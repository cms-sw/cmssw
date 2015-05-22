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
    def set(self,sv,ncorr): ## apply correction ncorr times
        self.var[0] = self.func(sv)
        if self.corrfunc:
            for i in range(ncorr):
                self.var[0] = self.corrfunc(self.var[0],sv.pt,sv.eta)
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
    def __call__(self,sv,ncorr): ## apply correction ncorr times
        for s in self.specs: s.set(sv,ncorr)
        for s in self.vars:  s.set(sv,ncorr)
        #return self.reader.EvaluateMVA(self.name)   
        return self.reader.GetRarity(self.name)   
class CategorizedMVA:
    def __init__(self,catMvaPairs):
        self.catMvaPairs = catMvaPairs
    def __call__(self,sv,ncorr):
        for c,m in self.catMvaPairs:
            if c(sv): return m(sv,ncorr)
        return -99.

_CommonSpect = [ 
]
_CommonVars = [ 
  
    MVAVar("SV_ntracks",lambda x: x.ntracks),  
    MVAVar("SV_mass",lambda x: x.mass),
    MVAVar("SV_ip2d := abs(SV_dxy)", lambda x : abs(x.dxy)),
    MVAVar("SV_sip2d := abs(SV_dxy/SV_edxy)", lambda x : abs(x.dxy/x.edxy)),
    MVAVar("SV_ip3d", lambda x : x.ip3d ),
    MVAVar("SV_sip3d", lambda x : x.sip3d ),
    MVAVar("SV_chi2n := min(SV_chi2/max(1,SV_ndof),10)",lambda x: min(x.chi2/max(1,x.ndof),10) ),
    MVAVar("SV_cosTheta := max(SV_cosTheta,0.98)",lambda x: max(x.cosTheta,0.98) ),
   
]

class SVMVA:
    def __init__(self,basepath):
        global _CommonVars, _CommonSpect
        self.svall = CategorizedMVA([
            ( lambda x: x.pt <= 15 and abs(x.eta) <  1.2 , MVATool("BDTG",basepath%"pteta_low_b", _CommonSpect,_CommonVars) ),
            ( lambda x: x.pt <= 15 and abs(x.eta) >= 1.2 , MVATool("BDTG",basepath%"pteta_low_e", _CommonSpect,_CommonVars) ),
            ( lambda x: x.pt >  15 and abs(x.eta) <  1.2 , MVATool("BDTG",basepath%"pteta_high_b",_CommonSpect,_CommonVars) ),
            ( lambda x: x.pt >  15 and abs(x.eta) >= 1.2 , MVATool("BDTG",basepath%"pteta_high_e",_CommonSpect,_CommonVars) ),
        ])       
    def __call__(self,sv,ncorr=0):
        return self.svall(sv,ncorr)

class SVMVAFriend:
    def __init__(self,path,fast=True):
        print path
        self.mva = SVMVA(path+"/weights/%s_BDTG.weights.xml")
        self.fast = fast
    def listBranches(self):
        return [ ("nSV","I"), ("SV_mva","F",20,"nSV"), ("nSV25_loose","I"), ("nSV25_medium","I"), ("nSV25_tight","I"), ("nSV25_stight","I")]
    def __call__(self,event):
        sv = Collection(event,"SV","nSV",20)
        ret = { 'nSV' : event.nSV }
        if event.run >= 1: # DATA
            ret['SV_mva'] = [ self.mva(s, ncorr=0) for s in sv ] 
        else:              # MC
            ret['SV_mva'] = [ self.mva(s, ncorr=0) for s in sv ] 
        ret['nSV25_loose'] = 0
        ret['nSV25_medium'] = 0
        ret['nSV25_tight'] = 0
        ret['nSV25_stight'] = 0
        for i,s in enumerate(sv):    
            if ret['SV_mva'][i] >= 0.3 and (s.jetPt < 25 or s.jetBTag<0.679): 
                ret['nSV25_loose'] += 1 
            if ret['SV_mva'][i] >= 0.7 and (s.jetPt < 25 or s.jetBTag<0.679):     
                ret['nSV25_medium'] += 1
            if ret['SV_mva'][i] >= 0.9 and (s.jetPt < 25 or s.jetBTag<0.679):     
                ret['nSV25_tight'] += 1
            if ret['SV_mva'][i] >= 0.96 and (s.jetPt < 25 or s.jetBTag<0.679):     
                ret['nSV25_stight'] += 1    
        return ret

if __name__ == '__main__':
    from sys import argv
    file = ROOT.TFile(argv[1])
    tree = file.Get("treeProducerSusyMultilepton")
    tree.vectorTree = True
    class Tester(Module):
        def __init__(self, name):
            Module.__init__(self,name,None)
            self.sf = SVMVAFriend("/afs/cern.ch/user/b/botta/CMGToolsGit/newRecipe70/CMSSW_7_0_6_patch1/src/CMGTools/TTHAnalysis/python/plotter/object-studies/")
        def analyze(self,ev):
            print "\nrun %6d lumi %4d event %d: leps %d" % (ev.run, ev.lumi, ev.evt, ev.nSV)
            print self.sf(ev)
    el = EventLoop([ Tester("tester") ])
    el.loop([tree], maxEvents = 50)

        
