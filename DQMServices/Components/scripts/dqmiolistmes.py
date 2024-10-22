#!/usr/bin/env python3
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
if 'Indices;1' in things:
  indices = f['Indices']
  runs = indices['Run'].array()
  lumis = indices['Lumi'].array()
  firstindex = indices['FirstIndex'].array()
  lastindex = indices['LastIndex'].array()
  types = indices['Type'].array()

  # If run is not provided, print all available runs in a given file.
  if args.r == None or args.r < 0:
    print('Please provide a valid run number. Runs contained in a given file:')
    print('To figure out which lumisections are available for each run, use dqmiodumpmetadata.py')
    for run in set(runs):
      print(run)
    exit()

  treenames = [ # order matters!
    "Ints",
    "Floats",
    "Strings",
    "TH1Fs",
    "TH1Ss",
    "TH1Ds",
    "TH2Fs",
    "TH2Ss",
    "TH2Ds",
    "TH3Fs",
    "TProfiles",
    "TProfile2Ds",
    "TH1Is",
    "TH2Is"
  ]
  trees = [f[name]["FullName"].array() for name in treenames]

  for run, lumi, first, last, type in zip(runs, lumis, firstindex, lastindex, types):
    if run == args.r and lumi == args.l:
      for i in range(first, int(last + 1)): # In DQMIO both ranges are inclusive
        print(trees[type][i])
else:
  print("This does not look like DQMIO data.")
