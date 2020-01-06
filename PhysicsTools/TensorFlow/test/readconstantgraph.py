# coding: utf-8

"""
Test script that reads the constant graph created by "createconstantgraph.py".
"""


import os
import sys
import tensorflow as tf

from PhysicsTools.TensorFlow.tools import TF2, read_constant_graph


# go into v1 compatibility mode
if TF2:
    tf = tf.compat.v1
tf.disable_eager_execution()

# prepare the datadir
if len(sys.argv) >= 2:
    datadir = sys.argv[1]
else:
    thisdir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(os.path.dirname(thisdir), "bin", "data")

# read and evaluate the graph
graph, sess = read_constant_graph(os.path.join(datadir, "constantgraph.pb"), create_session=True)
print(sess.run("output:0", feed_dict={"scale:0": 1.0, "input:0": [range(10)]})[0][0])
