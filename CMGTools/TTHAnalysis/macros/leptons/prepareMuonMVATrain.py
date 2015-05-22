#!/usr/bin/env python
from CMGTools.TTHAnalysis.treeReAnalyzer import *

import os,itertools, ROOT

from optparse import OptionParser
parser = OptionParser(usage="%prog [options] <PROCESS> <FILEOUT>")
parser.add_option("-t", "--tree",    dest="tree", default='ttHLepTreeProducerTTH', help="Pattern for tree name");
parser.add_option("-V", "--vector",  dest="vectorTree", action="store_true", default=True, help="Input tree is a vector")
parser.add_option("-b", "--b-only",  dest="backgroundOnly", action="store_true", default=False, help="Input tree is a vector")
(options, args) = parser.parse_args()
total = 0

class LepTreeProducer(Module):
    def beginJob(self):
        self.t = PyTree(self.book("TTree","t","t"))
        self.copyvars = [
            "pdgId",
            "mcMatchId",
            "mcMatchAny",
            "pt",
            "eta",
            "globalTrackChi2",
            "segmentCompatibility",
            "chi2LocalPosition",
            "chi2LocalMomentum",
            "innerTrackValidHitFraction",
            "lostOuterHits",
            "glbTrackProbability",
            "trackerHits",
            "lostHits",
            "tightId",
            "nStations",
            "trkKink",
            "caloCompatibility",
            "trackerLayers",
            "pixelLayers",
            "caloEMEnergy",
            "caloHadEnergy",
            "innerTrackChi2",
            "stationsWithAnyHits",
            "stationsWithValidHits",
            "stationsWithValidHitsGlbTrack",
        ]        
        for C in self.copyvars: self.t.branch("LepGood_"+C,"F")
        self.first = True
    def analyze(self,event):
        lep = Collection(event,"LepGood","nLepGood",8)
        global options, total
        for l in lep:
            if abs(l.pdgId) != 13: continue
            if options.backgroundOnly: 
                if l.mcMatchId > 0 or l.mcMatchAny == 2:
                    continue
            for C in self.copyvars: setattr(self.t, "LepGood_"+C, getattr(l,C))
            total += 1
            self.t.fill()

from sys import argv
f = ROOT.TFile.Open(args[0]+"/"+options.tree+"/"+options.tree+"_tree.root")
t = f.Get(options.tree)
t.vectorTree = options.vectorTree
print "Reading %s (%d entries)" % (args[0], t.GetEntries())

booker = Booker(args[1] if len(args) >= 2 else "lepTree.root")
prod = LepTreeProducer("rec",booker)
el = EventLoop([ prod, ])
maxEv = (int(args[3]) if len(args) >= 4 else -1)
print "max entries: ",maxEv
el.loop([t], maxEvents=maxEv)
booker.done()

print "Wrote %d entries" % total
