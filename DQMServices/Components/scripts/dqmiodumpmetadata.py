#!/usr/bin/env python
from __future__ import print_function

import uproot
import argparse

from collections import defaultdict

parser = argparse.ArgumentParser(description="Show which runs and lumisections are contained in a DQMIO file.")

parser.add_argument('filename', help='Name of local root file. For remote files, use edmCopyUtil first: `edmCopyUtil root://cms-xrd-global.cern.ch/<FILEPATH> .`')

args = parser.parse_args()

f = uproot.open(args.filename)
things = f.keys()
if b'Indices;1' in things:
  indices = f[b'Indices']
  runs = indices.array(b"Run")
  lumis = indices.array(b"Lumi")
  firstindex = indices.array(b"FirstIndex")
  lastindex = indices.array(b"LastIndex")
  types = indices.array(b"Type")

  counts = defaultdict(lambda: 0)
  for run, lumi, first, last, type in zip(runs, lumis, firstindex, lastindex, types):
    if type == 1000:
      n = 0
    else:
      n = last - first + 1
    counts[(run, lumi)] += n

  # Try to condense lumis into ranges for more compact output
  currentrun, currentcount = 0, 0
  minlumi, maxlumi = 0, 0

  def showrow():
    if currentrun != 0: # suppress first, empty row
      if (minlumi, maxlumi) == (0, 0): # run-based histos
        print("Run %d, %d MEs" % (currentrun, currentcount))
      else:
        print("Run %d, Lumi %d-%d, %d MEs" % (currentrun, minlumi, maxlumi, currentcount))
    
  for ((run, lumi), count) in sorted(counts.items()):
    if (currentrun, currentcount) != (run, count) or (lumi != maxlumi+1):
      showrow()
      minlumi, maxlumi = lumi, lumi
      currentrun, currentcount = run, count
    else:
      maxlumi = lumi # we assume order here
  showrow()

  print("Total: %d runs, %d lumisections." % (len([run for run, lumi in counts if lumi == 0]), len([lumi for run, lumi in counts if lumi != 0])))

else:
  print("This does not look like DQMIO data.")


