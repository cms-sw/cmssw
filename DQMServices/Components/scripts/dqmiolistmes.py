#!/usr/bin/env python
from __future__ import print_function

import uproot
import argparse

parser = argparse.ArgumentParser(description='List the full name of all MEs for a given run and lumi. ' +
                                             'If lumi is omitted, per run MEs will be printed out')

parser.add_argument('filename', help='Name of local root file. For remote files, use edmCopyUtil first: `edmCopyUtil root://cms-xrd-global.cern.ch/<FILEPATH> .`')
parser.add_argument('-r', type=int, help='Run to list MEs of')
parser.add_argument('-l', type=int, default=0, help='Lumisection to list MEs of')

args = parser.parse_args()

if args.l == None or args.l < 0:
  print('Please provide a valid lumisection number')
  exit()

f = uproot.open(args.filename)
things = f.keys()
if b'Indices;1' in things:
  indices = f[b'Indices']
  runs = indices.array(b"Run")
  lumis = indices.array(b"Lumi")
  firstindex = indices.array(b"FirstIndex")
  lastindex = indices.array(b"LastIndex")
  types = indices.array(b"Type")

  # If run is not provided, print all available runs in a given file.
  if args.r == None or args.r < 0:
    print('Please provide a valid run number. Runs contained in a given file:')
    print('To figure out which lumisections are available for each run, use dqmiodumpmetadata.py')
    for run in set(runs):
      print(run)
    exit()

  treenames = [ # order matters!
    b"Ints",
    b"Floats",
    b"Strings",
    b"TH1Fs",
    b"TH1Ss",
    b"TH1Ds",
    b"TH2Fs",
    b"TH2Ss",
    b"TH2Ds",
    b"TH3Fs",
    b"TProfiles",
    b"TProfile2Ds"
  ]
  trees = [f[name][b"FullName"].array() for name in treenames]

  for run, lumi, first, last, type in zip(runs, lumis, firstindex, lastindex, types):
    if run == args.r and lumi == args.l:
      for i in range(first, int(last + 1)): # In DQMIO both ranges are inclusive
        print(trees[type][i].decode())
else:
  print("This does not look like DQMIO data.")

