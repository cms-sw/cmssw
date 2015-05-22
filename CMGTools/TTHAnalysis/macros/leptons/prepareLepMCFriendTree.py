#!/usr/bin/env python
from CMGTools.TTHAnalysis.treeReAnalyzer import *
from array import array
from glob import glob
import os.path

import os, ROOT

def plausible(rec,gen):
    dr = deltaR(rec,gen)
    if abs(rec.pdgId) == 11 and abs(gen.pdgId) != 11:   return False
    if abs(rec.pdgId) == 13 and abs(gen.pdgId) != 13:   return False
    if dr < 0.3: return True
    if gen.pt < abs(rec.pdgId) == 13 and gen.pdgId != rec.pdgId: return False
    if dr < 0.7: return True
    if min(rec.pt,gen.pt)/max(rec.pt,gen.pt) < 0.3: return False
    return True
class LepMCTreeProducer(Module):
    def __init__(self,name,booker):
        Module.__init__(self,name,booker)
    def beginJob(self):
        self.t = PyTree(self.book("TTree","t","t"))
        self.e = 0
        for i in range(8):
            self.t.branch("LepGood%d_mcMatchNew" % (i+1),"F")
    def analyze(self,event):
        lep = Collection(event,"LepGood","nLepGood",8)
        glep = Collection(event,"GenLep")
        gtau = Collection(event,"GenLepFromTau")
        if event.evt in [ 265788744, ]:
            self.e = 0;
            print "Event %d" % event.evt
        else:
            self.e = 9999;
        
        for i,l in enumerate(lep):
            flag = 0
            (gmatch,dr) = closest(l,glep,presel=plausible)
            if dr < 1.5 and abs(gmatch.pdgId) == abs(l.pdgId):
                setattr(self.t, "LepGood%d_mcMatchNew" % (i+1), 2.)
                if self.e < 100: print "Lepton%d (%d,%.0f,%.2f,%.2f) -> match 2 with dr %.2f" % (i+1,l.pdgId,l.pt,l.eta,l.phi,dr)
            else:
                (gmatch,dr) = closest(l,gtau,presel=plausible)
                if dr < 1.5 and abs(gmatch.pdgId) == abs(l.pdgId):
                    setattr(self.t, "LepGood%d_mcMatchNew" % (i+1), 1.)
                    if self.e < 100: print "Lepton%d (%d,%.0f,%.2f,%.2f) -> match 1 with dr %.2f" % (i+1,l.pdgId,l.pt,l.eta,l.phi,dr)
                else:
                    setattr(self.t, "LepGood%d_mcMatchNew" % (i+1), 0.)
                    if self.e < 100: print "Lepton%d (%d,%.0f,%.2f,%.2f) -> match 0 with dr %.2f" % (i+1,l.pdgId,l.pt,l.eta,l.phi,dr)
        for i in xrange(len(lep),8):
            setattr(self.t, "LepGood%d_mcMatchNew" % (i+1), -99)
        self.t.fill()
        self.e += 1

import os, itertools
from sys import argv
if len(argv) < 3: print "Usage: %s <TREE_DIR> <TRAINING>" % argv[0]
jobs = []
if not os.path.exists(argv[2]):            os.mkdir(argv[2])
for D in glob(argv[1]+"/*"):
    fname = D+"/ttHLepTreeProducerBase/ttHLepTreeProducerBase_tree.root"
    if os.path.exists(fname):
        short = os.path.basename(D)
        data = ("DoubleMu" in short or "MuEG" in short or "DoubleElectron" in short)
        if data: continue
        f = ROOT.TFile.Open(fname);
        t = f.Get("ttHLepTreeProducerBase")
        entries = t.GetEntries()
        f.Close()
        chunk = 200000.
        if entries < chunk:
            print "  ",os.path.basename(D),("  DATA" if data else "  MC")," single chunk"
            jobs.append((short,fname,"%s/lepMCFriend_%s.root" % (argv[2],short),data,xrange(entries)))
        else:
            nchunk = int(ceil(entries/chunk))
            print "  ",os.path.basename(D),("  DATA" if data else "  MC")," %d chunks" % nchunk
            for i in xrange(nchunk):
                r = xrange(int(i*chunk),min(int((i+1)*chunk),entries))
                jobs.append((short,fname,"%s/lepMCFriend_%s.chunk%d.root" % (argv[2],short,i),data,r))
print 4*"\n"
print "I have %d taks to process" % len(jobs)

maintimer = ROOT.TStopwatch()
def _runIt(args):
    (name,fin,fout,data,range) = args
    timer = ROOT.TStopwatch()
    fb = ROOT.TFile(fin)
    tb = fb.Get("ttHLepTreeProducerBase")
    nev = tb.GetEntries()
    print "==== %s starting (%d entries) ====" % (name, nev)
    booker = Booker(fout)
    el = EventLoop([ LepMCTreeProducer("newMC",booker), ])
    el.loop([tb], eventRange=range)
    booker.done()
    fb.Close()
    time = timer.RealTime()
    print "=== %s done (%d entries, %.0f s, %.0f e/s) ====" % ( name, nev, time,(nev/time) )
    return (name,(nev,time))

from multiprocessing import Pool
pool = Pool(8)
ret  = dict(pool.map(_runIt, jobs))
fulltime = maintimer.RealTime()
totev   = sum([ev   for (ev,time) in ret.itervalues()])
tottime = sum([time for (ev,time) in ret.itervalues()])
print "Done %d tasks in %.1f min (%d entries, %.1f min)" % (len(jobs),fulltime/60.,totev,tottime/60.)

