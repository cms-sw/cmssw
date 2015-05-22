#!/usr/bin/env python
from CMGTools.TTHAnalysis.treeReAnalyzer import *
from array import array
from glob import glob
import os.path

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
  
    MVAVar("SV_ntracks",lambda x: x.numberOfDaughters()),  
    MVAVar("SV_mass",lambda x: x.mass()),
    MVAVar("SV_ip2d := abs(SV_dxy)", lambda x : abs(x.dxy.value())),
    MVAVar("SV_sip2d := abs(SV_dxy/SV_edxy)", lambda x : abs(x.dxy.value()/x.dxy.error())),
    MVAVar("SV_ip3d", lambda x : x.d3d.value() ),
    MVAVar("SV_sip3d", lambda x : x.d3d.significance() ),
    MVAVar("SV_chi2n := min(SV_chi2/max(1,SV_ndof),10)",lambda x: min(x.vertexChi2()/max(1,x.vertexNdof()),10) ),
    MVAVar("SV_cosTheta := max(SV_cosTheta,0.98)",lambda x: max(x.cosTheta,0.98) ),
   
]

class SVMVA:
    def __init__(self,basepath):
        global _CommonVars, _CommonSpect
        self.svall = CategorizedMVA([
            ( lambda x: x.pt() <= 15 and abs(x.eta()) <  1.2 , MVATool("BDTG",basepath%"pteta_low_b", _CommonSpect,_CommonVars) ),
            ( lambda x: x.pt() <= 15 and abs(x.eta()) >= 1.2 , MVATool("BDTG",basepath%"pteta_low_e", _CommonSpect,_CommonVars) ),
            ( lambda x: x.pt() >  15 and abs(x.eta()) <  1.2 , MVATool("BDTG",basepath%"pteta_high_b",_CommonSpect,_CommonVars) ),
            ( lambda x: x.pt() >  15 and abs(x.eta()) >= 1.2 , MVATool("BDTG",basepath%"pteta_high_e",_CommonSpect,_CommonVars) ),
        ])       
    def __call__(self,sv,ncorr=0):
        return self.svall(sv,ncorr)

