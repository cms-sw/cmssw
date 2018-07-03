#!/usr/bin/env python

# test for pytables
# taken from https://kastnerkyle.github.io/posts/using-pytables-for-larger-than-ram-data-processing/
# but with some interface modifications (presumably due to pytables changes)

import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tables

random_state = np.random.RandomState(1999)

def make_random_cluster_points(n_samples, random_state=random_state):
    mu_options = np.array([(-1, -1), (1, 1), (1, -1), (-1, 1)])
    sigma = 0.2
    mu_choices = random_state.randint(0, len(mu_options), size=n_samples)
    means = mu_options[mu_choices]
    return means + np.random.randn(n_samples, 2) * sigma, mu_choices

def plot_clusters(data, clusters, name):
    plt.figure()
    colors = ["#9b59b6", "#3498db", "#e74c3c", "#2ecc71"]
    for i in np.unique(clusters):
        plt.scatter(data[clusters==i, 0], data[clusters==i, 1], color=colors[i])
    plt.axis('off')
    plt.title('Plot from %s' % name)

sample_data, sample_clusters = make_random_cluster_points(10000)
hdf5_path = "my_data.hdf5"
hdf5_file = tables.file.open_file(hdf5_path, mode='w')
data_storage = hdf5_file.create_array(hdf5_file.root, 'data', sample_data)
clusters_storage = hdf5_file.create_array(hdf5_file.root, 'clusters', sample_clusters)
hdf5_file.close()

hdf5_path = "my_data.hdf5"
read_hdf5_file = tables.file.open_file(hdf5_path, mode='r')
# Here we slice [:] all the data back into memory, then operate on it
hdf5_data = read_hdf5_file.root.data[:]
hdf5_clusters = read_hdf5_file.root.clusters[:]
read_hdf5_file.close()

plot_clusters(hdf5_data, hdf5_clusters, "PyTables Array")
