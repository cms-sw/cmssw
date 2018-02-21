#!/usr/bin/env python
import sys
import ROOT
fin=ROOT.TFile.Open(sys.argv[1])
pu = fin.Get("pileup")
pileup = map( pu.GetBinContent, xrange(1,pu.GetNbinsX()+1) )
print ",".join(map(lambda x: "%1.3g"%x, pileup))
