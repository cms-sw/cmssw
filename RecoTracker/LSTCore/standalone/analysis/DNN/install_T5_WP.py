#!/usr/bin/env python3
"""Install a T5 DNN exported by wp_T5_DNN.py --export: copy the weights header to src/alpaka/T5NeuralNetworkWeights.h
and replace the kWp table (with its comment line) in namespace t5dnn of interface/alpaka/Common.h.
  python3 install_T5_WP.py --weights wp/sel_T5NeuralNetworkWeights.h --table wp/sel_kWp.h"""
import argparse
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--table", required=True, help="<out>_kWp.h written by wp_T5_DNN.py --export")
ap.add_argument("--common-h", default=os.path.join(HERE, "../../../interface/alpaka/Common.h"))
ap.add_argument("--weights-dst", default=os.path.join(HERE, "../../../src/alpaka/T5NeuralNetworkWeights.h"))
a = ap.parse_args()
s = open(a.common_h).read()
i = s.index("namespace t5dnn")
j = s.index("namespace", i + len("namespace t5dnn"))  # end of the t5dnn block (next namespace)
blk, n = re.subn(r"(      //[^\n]*\n)?      HOST_DEVICE_CONSTANT float kWp\[kPtBins\]\[kEtaBins\] = \{.*?\};\n",
                 lambda m: open(a.table).read(), s[i:j], count=1, flags=re.S)
assert n == 1, "kWp not found in namespace t5dnn"
open(a.common_h, "w").write(s[:i] + blk + s[j:])
shutil.copyfile(a.weights, a.weights_dst)
print("installed", a.weights, "and", a.table)
