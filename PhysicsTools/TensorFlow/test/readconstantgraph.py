# coding: utf-8

"""
Test script that reads the constant graph created by "createconstantgraph.py".
"""

import os
import sys

import cmsml


# get tensorflow and work with the v1 compatibility layer
tf, tf1, tf_version = cmsml.tensorflow.import_tf()
tf = tf1
tf.disable_eager_execution()

# prepare the datadir
if len(sys.argv) >= 2:
    datadir = sys.argv[1]
else:
    thisdir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(os.path.dirname(thisdir), "bin", "data")

# read and evaluate the graph
graph_path = os.path.join(datadir, "constantgraph.pb")
graph, sess = cmsml.tensorflow.load_graph(graph_path, create_session=True)
print(sess.run("output:0", feed_dict={"scale:0": 1.0, "input:0": [range(10)]})[0][0])
