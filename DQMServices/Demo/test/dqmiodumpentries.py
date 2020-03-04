#!/usr/bin/env python
import ROOT
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser(description="Output name and number of entries (or value) for MEs in a DQMIO file.")

parser.add_argument('inputfile', help='DQMIO ROOT file name.')
parser.add_argument('-r', '--run', help='Run number of run to process', default=1, type=int)
parser.add_argument('-l', '--lumi', help='Lumisection to process', default=0, type=int)
parser.add_argument('-s', '--summary', help='Only show values and how often they appeared.', action='store_true')
args = parser.parse_args()

treenames = {
    0: "Ints",
    1: "Floats",
    2: "Strings",
    3: "TH1Fs",
    4: "TH1Ss",
    5: "TH1Ds",
    6: "TH2Fs",
    7: "TH2Ss",
    8: "TH2Ds",
    9: "TH3Fs",
    10: "TProfiles",
    11: "TProfile2Ds",
}

f = ROOT.TFile.Open(args.inputfile)
idxtree = getattr(f, "Indices")

summary = defaultdict(lambda: 0)

for i in range(idxtree.GetEntries()):
    idxtree.GetEntry(i)
    run, lumi, metype = idxtree.Run, idxtree.Lumi, idxtree.Type
    if run != args.run or lumi != args.lumi:
        continue

    # inclusive range -- for 0 entries, row is left out
    firstidx, lastidx = idxtree.FirstIndex, idxtree.LastIndex
    metree = getattr(f, treenames[metype])
    # this GetEntry is only to make sure the TTree is initialized correctly
    metree.GetEntry(0)
    metree.SetBranchStatus("*",0)
    metree.SetBranchStatus("FullName",1)

    for x in range(firstidx, lastidx+1):
        metree.GetEntry(x)
        mename = str(metree.FullName)
        metree.GetEntry(x, 1)
        value = metree.Value
        
        
        if treenames[metype] in ["Ints", "Floats", "Strings"]:
          result = str(value)
        else:
          result = "%d" % value.GetEntries()

        if args.summary:
          summary[result] += 1
        else:
          print("%s: %s" % (mename, result))

if args.summary:
  keys = sorted(summary.keys())
  summaryitems = ["%s: %d" % (k, summary[k]) for k in keys]
  print(", ".join(summaryitems))
    




