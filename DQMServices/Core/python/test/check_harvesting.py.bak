#!/bin/env python

from ROOT import *
import sys
import re

RXOFFLINE = re.compile(r"^(?:.*/)?DQM_V(\d+)_R(\d+)((?:__[-A-Za-z0-9_]+){3})\.root$")
numEntries=50
histo_to_check = ['DQMData/Run %s/A_Folder/Run summary/Module/MyHisto',
                  'DQMData/Run %s/B_Folder/Run summary/Module/MyHisto']
means = [2.0, 3.0]

if len(sys.argv) < 1:
    print "Error, filename required\n"
    sys.exit(1)

filename = sys.argv[1]
m = re.match(RXOFFLINE, filename)
if not m:
    print "Error: wrong file supplied\n"
    sys.exit(1)

f = TFile(filename)

for h in range(0, len(histo_to_check)):
    print "Checking %s" % (histo_to_check[h] % str(int(m.group(2))))
    histo = f.Get(histo_to_check[h] % str(int(m.group(2))))
    assert(histo)
    assert(numEntries==histo.GetEntries())
    assert(means[h]==histo.GetMean())
    assert(0.0==histo.GetRMS())
sys.exit(0)
