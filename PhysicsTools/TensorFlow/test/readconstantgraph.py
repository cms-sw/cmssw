# -*- coding: utf-8 -*-

"""
Test script that reads the constant graph created by "createconstantgraph.py".
"""


import os
import sys
import tensorflow as tf
from tensorflow.python.platform import gfile


if len(sys.argv) >= 2:
    datadir = sys.argv[1]
else:
    thisdir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(os.path.dirname(thisdir), "bin", "data")


graph = tf.Graph()
with graph.as_default():
    graph_def = tf.GraphDef()
    with gfile.FastGFile(os.path.join(datadir, "constantgraph.pb"), "rb") as f:
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name="")

sess = tf.Session(graph=graph)

print(sess.run("output:0", feed_dict={"scale:0": 1.0, "input:0": [range(10)]})[0][0])
