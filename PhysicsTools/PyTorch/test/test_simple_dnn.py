#!/usr/bin/env python3

import sys
import os
import torch

if len(sys.argv)>=2:
    datadir=sys.argv[1]
else:
    thisdir=os.path.dirname(os.path.abspath(__file__))
    datadir=os.path.join(os.path.dirname(thisdir), "bin", "data")

ptfile = os.path.join(datadir, "simple_dnn.pt")
print("loading:", ptfile)

tm = torch.jit.load(ptfile)
tm.eval()

# dummy input (same shape used during trace: 10)
x = torch.ones(10)

# optional: run on gpu if available
device = "cuda" if torch.cuda.is_available() else "cpu"
tm.to(device)
x = x.to(device)

with torch.no_grad():
    y = tm(x)

print("ok. output:", y.item())
print("device:", device)
