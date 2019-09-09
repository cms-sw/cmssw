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
  # this is DQMIO data, TODO
  pass
elif 'DQMData;1' in things:
  basedir = f['DQMData']
  for run in basedir.keys():
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


