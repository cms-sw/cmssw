#!/usr/bin/env python
from CMGTools.TTHAnalysis.treeReAnalyzer import *
from array import array
from glob import glob
import os.path

import os, ROOT
if "/smearer_cc.so" not in ROOT.gSystem.GetLibraries(): 
    ROOT.gROOT.ProcessLine(".L %s/src/CMGTools/TTHAnalysis/python/plotter/smearer.cc+" % os.environ['CMSSW_BASE']);
if "/mcCorrections_cc.so" not in ROOT.gSystem.GetLibraries(): 
    ROOT.gROOT.ProcessLine(".L %s/src/CMGTools/TTHAnalysis/python/plotter/mcCorrections.cc+" % os.environ['CMSSW_BASE']);


class MVAVar:
    def __init__(self,name,func,corrfunc=None):
        self.name = name
        self.var  = array('f',[0.])
        self.func = func
        self.corrfunc = corrfunc
    def set(self,lep,ncorr): ## apply correction ncorr times
        self.var[0] = self.func(lep)
        if self.corrfunc:
            for i in range(ncorr):
                self.var[0] = self.corrfunc(self.var[0], lep.pdgId,lep.pt,lep.eta,lep.mcMatchId,lep.mcMatchAny)
class MVATool:
    def __init__(self,name,xml,specs,vars):
        self.name = name
        self.reader = ROOT.TMVA.Reader("Silent")
        self.specs = specs
        self.vars  = vars
        for s in specs: self.reader.AddSpectator(s.name,s.var)
        for v in vars:  self.reader.AddVariable(v.name,v.var)
        #print "Would like to load %s from %s! " % (name,xml)
        self.reader.BookMVA(name,xml)
    def __call__(self,lep,ncorr): ## apply correction ncorr times
        for s in self.specs: s.set(lep,ncorr)
        for s in self.vars:  s.set(lep,ncorr)
        return self.reader.EvaluateMVA(self.name)   
class CategorizedMVA:
    def __init__(self,catMvaPairs):
        self.catMvaPairs = catMvaPairs
    def __call__(self,lep,ncorr):
        for c,m in self.catMvaPairs:
            if c(lep): return m(lep,ncorr)
        return -99.

_CommonSpect = [ 
]
_CommonVars = [ 
    MVAVar("neuRelIso03 := relIso03 - chargedHadRelIso03",lambda x: x.relIso03 - x.chargedHadRelIso03),  
    MVAVar("chRelIso03 := chargedHadRelIso03",lambda x: x.chargedHadRelIso03),
    MVAVar("jetDR := min(jetDR,0.5)", lambda x : min(x.jetDR,0.5), corrfunc=ROOT.correctJetDRMC),
    MVAVar("jetPtRatio := min(jetPtRatio,1.5)", lambda x : min(x.jetPtRatio,1.5), corrfunc=ROOT.correctJetPtRatioMC),
    MVAVar("jetBTagCSV := max(jetBTagCSV,0)", lambda x : max(x.jetBTagCSV,0.)),
    MVAVar("sip3d",lambda x: x.sip3d, corrfunc=ROOT.scaleSip3dMC),
    MVAVar("dxy := log(abs(dxy))",lambda x: log(abs(x.dxy)), corrfunc=ROOT.scaleDxyMC),
    MVAVar("dz  := log(abs(dz))", lambda x: log(abs(x.dz)), corrfunc=ROOT.scaleDzMC),
]
_ElectronVars = [
    MVAVar("mvaId",lambda x: x.mvaId)
]
class LeptonMVA:
    def __init__(self,basepath):
        global _CommonVars, _CommonSpect, _ElectronVars
        self.mu = CategorizedMVA([
            ( lambda x: x.pt <= 15 and abs(x.eta) <  1.5 , MVATool("BDTG",basepath%"mu_pteta_low_b", _CommonSpect,_CommonVars) ),
            ( lambda x: x.pt <= 15 and abs(x.eta) >= 1.5 , MVATool("BDTG",basepath%"mu_pteta_low_e", _CommonSpect,_CommonVars) ),
            ( lambda x: x.pt >  15 and abs(x.eta) <  1.5 , MVATool("BDTG",basepath%"mu_pteta_high_b",_CommonSpect,_CommonVars) ),
            ( lambda x: x.pt >  15 and abs(x.eta) >= 1.5 , MVATool("BDTG",basepath%"mu_pteta_high_e",_CommonSpect,_CommonVars) ),
        ])
        self.el = CategorizedMVA([
            ( lambda x: x.pt <= 10 and abs(x.eta) <  0.8                         , MVATool("BDTG",basepath%"el_pteta_low_cb", _CommonSpect,_CommonVars+_ElectronVars) ),
            ( lambda x: x.pt <= 10 and abs(x.eta) >= 0.8 and abs(x.eta) <  1.479 , MVATool("BDTG",basepath%"el_pteta_low_fb", _CommonSpect,_CommonVars+_ElectronVars) ),
            ( lambda x: x.pt <= 10 and abs(x.eta) >= 1.479                       , MVATool("BDTG",basepath%"el_pteta_low_ec", _CommonSpect,_CommonVars+_ElectronVars) ),
            ( lambda x: x.pt >  10 and abs(x.eta) <  0.8                         , MVATool("BDTG",basepath%"el_pteta_high_cb",_CommonSpect,_CommonVars+_ElectronVars) ),
            ( lambda x: x.pt >  10 and abs(x.eta) >= 0.8 and abs(x.eta) <  1.479 , MVATool("BDTG",basepath%"el_pteta_high_fb",_CommonSpect,_CommonVars+_ElectronVars) ),
            ( lambda x: x.pt >  10 and abs(x.eta) >= 1.479                       , MVATool("BDTG",basepath%"el_pteta_high_ec",_CommonSpect,_CommonVars+_ElectronVars) ),
        ])
    def __call__(self,lep,ncorr=0):
        if   abs(lep.pdgId) == 11: return self.el(lep,ncorr)
        elif abs(lep.pdgId) == 13: return self.mu(lep,ncorr)
        else: return -99

class LepMVATreeProducer(Module):
    def __init__(self,name,booker,path,data,fast=False):
        Module.__init__(self,name,booker)
        self.mva = LeptonMVA(path+"/weights/%s_BDTG.weights.xml")
        self.data = data
        self.fast = fast
    def beginJob(self):
        self.t = PyTree(self.book("TTree","t","t"))
        for i in range(8):
            self.t.branch("LepGood%d_mvaNew" % (i+1),"F")
            #if not self.data and not self.fast:
            #    self.t.branch("LepGood%d_mvaNewUncorr"     % (i+1),"F")
            #    self.t.branch("LepGood%d_mvaNewDoubleCorr" % (i+1),"F")
    def analyze(self,event):
        lep = Collection(event,"LepGood","nLepGood",8)
        for i,l in enumerate(lep):
            if self.data:
                setattr(self.t, "LepGood%d_mvaNew" % (i+1), self.mva(l,ncorr=0))
            else: 
                setattr(self.t, "LepGood%d_mvaNew" % (i+1), self.mva(l,ncorr=0))
                #if not self.fast:
                #    setattr(self.t, "LepGood%d_mvaNewUncorr"     % (i+1), self.mva(l,ncorr=0))
                #    setattr(self.t, "LepGood%d_mvaNewDoubleCorr" % (i+1), self.mva(l,ncorr=2))
        for i in xrange(len(lep),8):
            setattr(self.t, "LepGood%d_mvaNew" % (i+1), -99.)
            #if not self.data and not self.fast:
            #    setattr(self.t, "LepGood%d_mvaNewUncorr"     % (i+1), -99.)
            #    setattr(self.t, "LepGood%d_mvaNewDoubleCorr" % (i+1), -99.)
        self.t.fill()

import os, itertools
from sys import argv
if len(argv) < 3: print "Usage: %s [-e] <TREE_DIR> <TRAINING>" % argv[0]
jobs = []
if argv[1] == "-e": # use existing training
    argv = [argv[0]] + argv[2:]
else: # new, make directory training
    if not os.path.exists(argv[2]):            os.mkdir(argv[2])
    if not os.path.exists(argv[2]+"/weights"): os.mkdir(argv[2]+"/weights")
    if not os.path.exists(argv[2]+"/trainings"): os.mkdir(argv[2]+"/trainings")
    for X,Y in itertools.product(["high","low"],["b","e"]):
        os.system("cp weights/mu_pteta_%s_%s_BDTG.weights.xml %s/weights -v" % (X,Y,argv[2]))
        os.system("cp mu_pteta_%s_%s.root %s/trainings -v" % (X,Y,argv[2]))
    for X,Y in itertools.product(["high","low"],["cb","fb","ec"]):
        os.system("cp weights/el_pteta_%s_%s_BDTG.weights.xml %s/weights -v" % (X,Y,argv[2]))
        os.system("cp el_pteta_%s_%s.root %s/trainings -v" % (X,Y,argv[2]))
for D in glob(argv[1]+"/*"):
    fname = D+"/ttHLepTreeProducerBase/ttHLepTreeProducerBase_tree.root"
    if os.path.exists(fname):
        short = os.path.basename(D)
        data = ("DoubleMu" in short or "MuEG" in short or "DoubleElectron" in short)
        f = ROOT.TFile.Open(fname);
        t = f.Get("ttHLepTreeProducerBase")
        entries = t.GetEntries()
        f.Close()
        chunk = 500000.
        if entries < chunk:
            print "  ",os.path.basename(D),("  DATA" if data else "  MC")," single chunk"
            jobs.append((short,fname,"%s/lepMVAFriend_%s.root" % (argv[2],short),data,xrange(entries)))
        else:
            nchunk = int(ceil(entries/chunk))
            print "  ",os.path.basename(D),("  DATA" if data else "  MC")," %d chunks" % nchunk
            for i in xrange(nchunk):
                r = xrange(int(i*chunk),min(int((i+1)*chunk),entries))
                jobs.append((short,fname,"%s/lepMVAFriend_%s.chunk%d.root" % (argv[2],short,i),data,r))
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
    el = EventLoop([ LepMVATreeProducer("newMVA",booker,argv[2],data,fast=True), ])
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

