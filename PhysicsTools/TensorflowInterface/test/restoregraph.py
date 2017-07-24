# -*- coding: utf-8 -*-

"""
Simple test to restore and evaluate the graph produced by creategraph.py and saved at bin/data/.
"""


import os
import tensorflow as tf


thisdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(os.path.dirname(thisdir), "bin", "data")

sess = tf.Session()

saver = tf.train.import_meta_graph(os.path.join(datadir, "simplegraph.meta"))
saver.restore(sess, os.path.join(datadir, "simplegraph"))

print(sess.run({"y": "output:0"}, feed_dict={"input:0": [range(10)]})["y"][0][0])
