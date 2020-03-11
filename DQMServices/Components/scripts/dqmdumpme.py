#!/usr/bin/env python
from __future__ import print_function
import sys
import numpy
import uproot
import argparse

numpy.set_printoptions(threshold=sys.maxsize)

parser = argparse.ArgumentParser(description="Display in text format a single ME out of a DQM (legacy or DQMIO) file. " +
                                             "If there is more than on copy, of one ME, all are shown.")

parser.add_argument('filename', help='Name of local root file. For remote files, use edmCopyUtil first: `edmCopyUtil root://cms-xrd-global.cern.ch/<FILEPATH> .`')
parser.add_argument('mepaths', metavar='ME', nargs='+', help='Path of ME to extract.')

args = parser.parse_args()

f = uproot.open(args.filename)
things = f.keys()
if 'Indices;1' in things:
  # this is DQMIO data

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
    "TProfile2Ds"
  ]
  trees = [{"FullName": f[name]["FullName"].array(), "Value": f[name]["Value"].lazyarray()} for name in treenames]

  def binsearch(a, key, lower, upper):
    n = upper - lower
    if n <= 1: return lower
    mid = int(n / 2) + lower
    if a[mid] < key: return binsearch(a, key, mid, upper)
    else: return binsearch(a, key, lower, mid)
  def linsearch(a, key, lower, upper):
    for k in range(lower, upper):
      if a[k] == key: return k
    return 0

  indices = f['Indices'].lazyarrays()
  for idx in range(len(indices["Run"])):
    run = indices["Run"][idx]
    lumi = indices["Lumi"][idx]
    type = indices["Type"][idx]
    first = int(indices["FirstIndex"][idx])
    last = int(indices["LastIndex"][idx])
    if type == 1000: continue # no MEs here
    names = trees[type]["FullName"]
    for me in args.mepaths:
      k = linsearch(names, me, first, last+1)
      if names[k] == me:
        meobj = trees[type]["Value"][k]
        print("ME for run %d, lumi %d" % (run, lumi), meobj)
        # uproot can't read most TH1 types from trees right now.

elif 'DQMData;1' in things:
  basedir = f['DQMData']
  for run in basedir.keys():
    if not run.startswith("Run "): continue
    rundir = basedir[run]
    print("MEs for %s" % run)
    for me in args.mepaths:
      subsys, path = me.split('/', 1)
      subsysdir = rundir[subsys]
      # typically this will only be "Run summary"
      for lumi in subsysdir.keys():
        print("  MEs for %s" % lumi)
        lumidir = subsysdir[lumi]
        meobj = lumidir[path]
        try:
          print(meobj.show())
        except:
          print(meobj.numpy().__str__())


