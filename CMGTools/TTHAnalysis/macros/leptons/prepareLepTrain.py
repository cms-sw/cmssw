#!/usr/bin/env python
from CMGTools.TTHAnalysis.treeReAnalyzer import *

import os, os.path, itertools, ROOT
if "/smearer_cc.so" not in ROOT.gSystem.GetLibraries(): 
    ROOT.gROOT.ProcessLine(".L %s/src/CMGTools/TTHAnalysis/python/plotter/smearer.cc+" % os.environ['CMSSW_BASE']);
if "/mcCorrections_cc.so" not in ROOT.gSystem.GetLibraries(): 
    ROOT.gROOT.ProcessLine(".L %s/src/CMGTools/TTHAnalysis/python/plotter/mcCorrections.cc+" % os.environ['CMSSW_BASE']);


from optparse import OptionParser
parser = OptionParser(usage="%prog [options] <PROCESS> <FILEOUT>")
parser.add_option("-t", "--tree",    dest="tree", default='ttHLepTreeProducerTTH', help="Pattern for tree name");
parser.add_option("-V", "--vector",  dest="vectorTree", action="store_true", default=True, help="Input tree is a vector")
parser.add_option("-F", "--add-friend",    dest="friendTrees",  action="append", default=[], nargs=2, help="Add a friend tree (treename, filename). Can use {name}, {cname} patterns in the treename") 
parser.add_option("-N", "--events",  dest="chunkSize", type="int",    default=500000, help="Default chunk size when splitting trees");
parser.add_option("-c", "--chunk",   dest="chunks",    type="int",    default=[], action="append", help="Process only these chunks (works only if a single dataset is selected with -d)");
parser.add_option("-j", "--jobs",    dest="jobs",      type="int",    default=0, help="Use N threads");
(options, args) = parser.parse_args()

class LepTreeProducer(Module):
    def beginJob(self):
        self.t = PyTree(self.book("TTree","t","t"))
        self.t.branch("good","I")
        self.t.branch("nJet25","I")
        self.t.branch("nBJetLoose25","I")
        self.t.branch("nBJetMedium25","I")
        self.t.branch("puWeight","F")
        self.copyvars = ['relIso03','chargedHadRelIso03','relIso04','chargedHadRelIso04','pt','eta','pdgId','lostHits','tightId','nStations','trkKink','caloCompatibility','globalTrackChi2','trackerLayers','pixelLayers','mcMatchId','mcMatchAny', 'mcMatchTau']        
        self.copyvars += [ 'hasSV', 'svRedPt', 'svRedM', 'svLepSip3d', 'svSip3d', 'svNTracks', 'svChi2n', 'svDxy', 'svM', 'svPt', ]
        for C in self.copyvars: self.t.branch(C,"F")
        # these I can't copy since I need to apply corrections
        for C in [ 'sip3d','dxy','dz','jetPtRatio','jetBTagCSV','jetDR','mvaId']: self.t.branch(C,"F")
        self.first = True
    def analyze(self,event):
        lep = Collection(event,"LepGood","nLepGood",8)
        for l in lep:
            self.t.sip3d = ROOT.scaleSip3dMC(l.sip3d, l.pdgId,l.pt,l.eta,l.mcMatchId,l.mcMatchAny) if self.corr else l.sip3d
            self.t.dz    = ROOT.scaleDzMC(   l.dz,    l.pdgId,l.pt,l.eta,l.mcMatchId,l.mcMatchAny) if self.corr else l.dz
            self.t.dxy   = ROOT.scaleDxyMC(  l.dxy,   l.pdgId,l.pt,l.eta,l.mcMatchId,l.mcMatchAny) if self.corr else l.dxy
            (dr,ptf) = (l.jetDR,l.jetPtRatio)
            self.t.jetDR  = ROOT.correctJetDRMC(dr,l.pdgId,l.pt,l.eta,l.mcMatchId,l.mcMatchAny)       if self.corr else dr
            self.t.jetPtRatio = ROOT.correctJetPtRatioMC(ptf,l.pdgId,l.pt,l.eta,l.mcMatchId,l.mcMatchAny) if self.corr else ptf
            self.t.jetBTagCSV = l.jetBTagCSV
            self.t.mvaId = l.mvaId if abs(l.pdgId) == 11 else l.muonMVAIdFull
            for C in self.copyvars: setattr(self.t, C, getattr(l,C))
            self.t.nJet25 = event.nJet25
            self.t.nBJetLoose25 = event.nBJetLoose25
            self.t.nBJetMedium25 = event.nBJetMedium25
            self.t.puWeight = event.puWeight
            self.t.fill()

f = ROOT.TFile.Open(args[0]+"/"+options.tree+"/"+options.tree+"_tree.root")
t = f.Get(options.tree)
nchunks = int(ceil(t.GetEntries()/float(options.chunkSize)))
print "Reading %s (%d entries, %d chunks)" % (args[0], t.GetEntries(), nchunks)
f.Close()

def run(ichunk):
    global args, options, nchunks
    if options.chunks and (ichunk not in options.chunks): return None
    print "Processing %s, chunk %d" % (args[0], ichunk)
    f = ROOT.TFile.Open(args[0]+"/"+options.tree+"/"+options.tree+"_tree.root")
    t = f.Get(options.tree)
    t.vectorTree = options.vectorTree
    friends_ = [] # to make sure pyroot does not delete them
    for tf_tree,tf_file in options.friendTrees:
        tf = t.AddFriend(tf_tree, tf_file.format(name=os.path.basename(args[0]), cname=os.path.basename(args[0]))),
        friends_.append(tf) # to make sure pyroot does not delete them
    fname = args[1] if len(args) >= 2 else "lepTree.root"
    if nchunks > 1: fname  = fname.replace(".root","")+(".chunk%d.root" % ichunk)
    booker = Booker(fname)
    prod = LepTreeProducer("rec",booker)
    if len(args) >= 3 and args[2] == "NoCorr": 
        print "Will not apply corrections"
        prod.corr = False
    else:
        print "Will apply corrections"
        prod.corr = True
    el = EventLoop([ prod, ])
    r = xrange(int(ichunk*options.chunkSize),min(int((ichunk+1)*options.chunkSize),t.GetEntries()))
    el.loop([t], eventRange=r)
    booker.done()
    f.Close()

if options.jobs > 0:
    from multiprocessing import Pool
    pool = Pool(options.jobs)
    pool.map(run, xrange(nchunks))
else:
    for ichunk in xrange(nchunks): run(ichunk)

