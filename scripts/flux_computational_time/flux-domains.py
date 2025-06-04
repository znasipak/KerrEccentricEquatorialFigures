import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from seaborn import color_palette
from few.utils.mappings.kerrecceq import apex_of_uwyz, apex_of_UWYZ

# use computer modern font
plt.rcParams.update({'font.size': 14})
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern"]

# seaborn colorblind palette
cpal = color_palette("colorblind")

if __name__ == "__main__":
    
    # build inner grids
    uarr = np.linspace(0, 1, 33)
    warr = np.linspace(0, 1, 33)
    zarr = np.linspace(0, 1, 65)

    ugrid, wgrid, zgrid = np.meshgrid(uarr, warr, zarr, indexing='ij')
    agrid, pgrid, egrid, xgrid = apex_of_uwyz(ugrid.flatten(), wgrid.flatten(), 1, zgrid.flatten())
    agrid = agrid.reshape(ugrid.shape)
    pgrid = pgrid.reshape(ugrid.shape)
    egrid = egrid.reshape(ugrid.shape)

    # build outer grids
    Uarr = np.linspace(0, 1, 33)
    Warr = np.linspace(0, 1, 33)
    Zarr = np.linspace(0, 1, 33)

    Ugrid, Wgrid, Zgrid = np.meshgrid(Uarr, Warr, Zarr, indexing='ij')
    Agrid, Pgrid, Egrid, Zgrid = apex_of_UWYZ(Ugrid.flatten(), Wgrid.flatten(), 1, Zgrid.flatten(), is_flux=True)
    Agrid = Agrid.reshape(Ugrid.shape)
    Pgrid = Pgrid.reshape(Ugrid.shape)
    Egrid = Egrid.reshape(Ugrid.shape)

    # Choose z / a-slices to plot
    a_iter_list = [0, 18, 36, 50, 64] # must be even!
    
    # Set up figure orientation
    fig_num = 2
    aspect_ratio = 1

    # Create subplots
    # fig, ax = plt.subplots(1, fig_num, figsize=(fig_num*6, 6*aspect_ratio), sharey=True)
    fig, ax = plt.subplots(fig_num, 1, figsize=(6, fig_num*6*aspect_ratio), sharey=True)
    ax = ax.flatten()

    # Iterate over different slices
    i = 0
    for i, a_iter in enumerate(a_iter_list):
        
        # Get the value of a
        aval = agrid[0,0,a_iter]
        # Take this slice of the data with constant aval
        p_data = pgrid[:,:,a_iter]
        e_data = egrid[:,:,a_iter]
        
        # Generate scatter plots
        my_cmap = sns.light_palette(cpal[i], as_cmap=True)
        p_max_vals = np.concatenate((p_data[0, :], p_data[:, -1]))
        e_max_vals = np.concatenate((e_data[0, :], e_data[:, -1]))
        p_min_vals = np.concatenate((p_data[:, 0], p_data[-1, :]))
        e_min_vals = np.concatenate((e_data[:, 0], e_data[-1, :]))
        ax[0].plot(p_data[:, -1], e_data[:, -1], color=cpal[i], label = f'$a={aval:0.3f}$', rasterized=True)
        ax[0].plot(p_data[:, 0], e_data[:, 0], color=cpal[i], rasterized=True)
        ax[0].plot(p_data[0, :], e_data[0, :], color=cpal[i], rasterized=True)
        ax[0].plot(p_data[-1, :], e_data[-1, :], color=cpal[i], rasterized=True)
        xdata = np.concatenate((p_max_vals, np.flip(p_min_vals)))
        ydata = np.concatenate((e_max_vals, np.flip(e_min_vals)))
        ax[0].fill(xdata, ydata, facecolor=cpal[i], edgecolor=cpal[i], alpha=0.2, rasterized=True)
        ax[0].fill(xdata, ydata, facecolor='none', edgecolor=cpal[i], rasterized=True)

        # Outer grid data
        a_iter_outer = a_iter // 2 # Outer grid has half the resolution in a
        
        # double check that we are taking the same aval slice for the outer and inner grids
        aval_outer = Agrid[0,0,a_iter_outer]
        if aval_outer != aval:
            raise ValueError("Mismatch in aval")
        
        # Take this slice of the outer data with constant aval
        p_data = Pgrid[:,:,a_iter_outer]
        e_data = Egrid[:,:,a_iter_outer]
        
        p_max_vals = np.concatenate((p_data[0, :], p_data[:, -1]))
        e_max_vals = np.concatenate((e_data[0, :], e_data[:, -1]))
        p_min_vals = np.concatenate((p_data[:, 0], p_data[-1, :]))
        e_min_vals = np.concatenate((e_data[:, 0], e_data[-1, :]))
        # ax[1].plot(p_data[:, -1], e_data[:, -1], color=cpal[i], label = f'$a={aval:0.3f}$', rasterized=True)
        # ax[1].plot(p_data[:, 0], e_data[:, 0], color=cpal[i], rasterized=True)
        # ax[1].plot(p_data[0, :], e_data[0, :], color=cpal[i], rasterized=True)
        # ax[1].plot(p_data[-1, :], e_data[-1, :], color=cpal[i], rasterized=True)
        xdata = np.concatenate((p_max_vals, np.flip(p_min_vals)))
        ydata = np.concatenate((e_max_vals, np.flip(e_min_vals)))
        ax[1].fill(xdata, ydata, facecolor=cpal[i], edgecolor=cpal[i], alpha=0.2, rasterized=True)
        ax[1].fill(xdata, ydata, facecolor='none', edgecolor=cpal[i], rasterized=True)

        ax[0].set_xlabel('$p$')
        ax[1].set_xlabel('$p$')
        ax[0].set_ylabel('$e$')
        ax[1].set_ylabel('$e$')

    plt.ylim(-0.05, 1.05)
    ax[0].legend(loc = 'upper left', fontsize = 10)

    # Show/save plot
    plt.savefig('data_domains.pdf', dpi=300, bbox_inches='tight')