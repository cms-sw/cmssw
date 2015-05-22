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


# _CommonVars = [ 
#     MVAVar("neuRelIso := relIso - chargedIso/pt",lambda x: x.relIso - x.chargedIso/x.pt),  
#     MVAVar("chRelIso := chargedIso/pt",lambda x: x.chargedIso/x.pt),
#     MVAVar("jetDR_in := min(dr_in,0.5)", lambda x : min(x.jetDR,0.5), corrfunc=ROOT.correctJetDRMC),
#     MVAVar("jetPtRatio_in := min(ptf_in,1.5)", lambda x : min(x.jetPtRatio,1.5), corrfunc=ROOT.correctJetPtRatioMC),
#     MVAVar("jetBTagCSV_in := max(CSV_in,0)", lambda x : max(x.jetBTagCSV,0.)),
#     #MVAVar("jetDR_out := min(dr_out,5)", lambda x : min(x.dr_out,5.)),
#     #MVAVar("jetPtRatio_out := min(ptf_out,1.5)", lambda x : min(x.ptf_out,1.5)),
#     #MVAVar("jetBTagCSV_out := max(CSV_out,0)", lambda x : max(x.CSV_out,0.)),
#     MVAVar("sip3d",lambda x: x.sip3d, corrfunc=ROOT.scaleSip3dMC),
#     MVAVar("dxy := log(abs(dxy))",lambda x: log(abs(x.dxy)), corrfunc=ROOT.scaleDxyMC),
#     MVAVar("dz  := log(abs(dz))", lambda x: log(abs(x.dz)), corrfunc=ROOT.scaleDzMC),
# ]
# _ElectronVars = [
#     MVAVar("mvaId",lambda x: x.mvaId),
#     MVAVar("innerHits",lambda x: x.innerHits),
# ]


_IsoVars = [
    MVAVar("neuRelIso := relIso - chargedIso/pt",lambda x: x.relIso - x.chargedIso/x.pt),  
    MVAVar("chRelIso := chargedIso/pt",lambda x: x.chargedIso/x.pt),
]

_JetVars = [
    MVAVar("jetDR_in := min(dr_in,0.5)", lambda x : min(x.jetDR,0.5), corrfunc=ROOT.correctJetDRMC),
    MVAVar("jetPtRatio_in := min(ptf_in,1.5)", lambda x : min(x.jetPtRatio,1.5), corrfunc=ROOT.correctJetPtRatioMC),

]    


_SIPVars = [
    MVAVar("sip3d",lambda x: x.sip3d, corrfunc=ROOT.scaleSip3dMC),
 
]

_IpVars = [
    MVAVar("dxy := log(abs(dxy))",lambda x: log(abs(x.dxy)), corrfunc=ROOT.scaleDxyMC),
    MVAVar("dz  := log(abs(dz))", lambda x: log(abs(x.dz)), corrfunc=ROOT.scaleDzMC),

]


_BTagVars = [
    MVAVar("jetBTagCSV_in := max(CSV_in,0)", lambda x : max(x.jetBTagCSV,0.)),

]


_MvaIdVars = [
    MVAVar("mvaId",lambda x: x.mvaId), 
]


_InnerHitsVars = [
    MVAVar("innerHits",lambda x: x.innerHits), 
]


class LeptonMVA:
    def __init__(self,basepath):
        global _CommonVars, _CommonSpect, _ElectronVars
        self.mu = CategorizedMVA([
            ( lambda x: x.pt <= 15 and abs(x.eta) <  1.5 , MVATool("BDTG",basepath%"mu_pteta_low_b", _CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_IpVars) ),
            ( lambda x: x.pt <= 15 and abs(x.eta) >= 1.5 , MVATool("BDTG",basepath%"mu_pteta_low_e", _CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_IpVars) ),
            ( lambda x: x.pt >  15 and abs(x.eta) <  1.5 , MVATool("BDTG",basepath%"mu_pteta_high_b",_CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_IpVars) ),
            ( lambda x: x.pt >  15 and abs(x.eta) >= 1.5 , MVATool("BDTG",basepath%"mu_pteta_high_e",_CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_IpVars) ),
        ])
        self.el = CategorizedMVA([
            ( lambda x: x.pt <= 10 and abs(x.eta) <  0.8                         , MVATool("BDTG",basepath%"el_pteta_low_cb", _CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_IpVars+_MvaIdVars+_InnerHitsVars) ),
            ( lambda x: x.pt <= 10 and abs(x.eta) >= 0.8 and abs(x.eta) <  1.479 , MVATool("BDTG",basepath%"el_pteta_low_fb", _CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_IpVars+_MvaIdVars+_InnerHitsVars) ),
            ( lambda x: x.pt <= 10 and abs(x.eta) >= 1.479                       , MVATool("BDTG",basepath%"el_pteta_low_ec", _CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_IpVars+_MvaIdVars+_InnerHitsVars) ),
            ( lambda x: x.pt >  10 and abs(x.eta) <  0.8                         , MVATool("BDTG",basepath%"el_pteta_high_cb",_CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_IpVars+_MvaIdVars+_InnerHitsVars) ),
            ( lambda x: x.pt >  10 and abs(x.eta) >= 0.8 and abs(x.eta) <  1.479 , MVATool("BDTG",basepath%"el_pteta_high_fb",_CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_IpVars+_MvaIdVars+_InnerHitsVars) ),
            ( lambda x: x.pt >  10 and abs(x.eta) >= 1.479                       , MVATool("BDTG",basepath%"el_pteta_high_ec",_CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_IpVars+_MvaIdVars+_InnerHitsVars) ),
        ])
    def __call__(self,lep,ncorr=0):
        if   abs(lep.pdgId) == 11: return self.el(lep,ncorr)
        elif abs(lep.pdgId) == 13: return self.mu(lep,ncorr)
        else: return -99

#=========


class LeptonMVANoIso:
    def __init__(self,basepath):
        global _CommonVars, _CommonSpect, _ElectronVars
        self.mu = CategorizedMVA([
            ( lambda x: x.pt <= 15 and abs(x.eta) <  1.5 , MVATool("BDTG",basepath%"mu_pteta_low_b", _CommonSpect,_JetVars+_BTagVars+_SIPVars+_IpVars) ),
            ( lambda x: x.pt <= 15 and abs(x.eta) >= 1.5 , MVATool("BDTG",basepath%"mu_pteta_low_e", _CommonSpect,_JetVars+_BTagVars+_SIPVars+_IpVars) ),
            ( lambda x: x.pt >  15 and abs(x.eta) <  1.5 , MVATool("BDTG",basepath%"mu_pteta_high_b",_CommonSpect,_JetVars+_BTagVars+_SIPVars+_IpVars) ),
            ( lambda x: x.pt >  15 and abs(x.eta) >= 1.5 , MVATool("BDTG",basepath%"mu_pteta_high_e",_CommonSpect,_JetVars+_BTagVars+_SIPVars+_IpVars) ),
        ])
        self.el = CategorizedMVA([
            ( lambda x: x.pt <= 10 and abs(x.eta) <  0.8                         , MVATool("BDTG",basepath%"el_pteta_low_cb", _CommonSpect,_JetVars+_BTagVars+_SIPVars+_IpVars+_MvaIdVars+_InnerHitsVars) ),
            ( lambda x: x.pt <= 10 and abs(x.eta) >= 0.8 and abs(x.eta) <  1.479 , MVATool("BDTG",basepath%"el_pteta_low_fb", _CommonSpect,_JetVars+_BTagVars+_SIPVars+_IpVars+_MvaIdVars+_InnerHitsVars) ),
            ( lambda x: x.pt <= 10 and abs(x.eta) >= 1.479                       , MVATool("BDTG",basepath%"el_pteta_low_ec", _CommonSpect,_JetVars+_BTagVars+_SIPVars+_IpVars+_MvaIdVars+_InnerHitsVars) ),
            ( lambda x: x.pt >  10 and abs(x.eta) <  0.8                         , MVATool("BDTG",basepath%"el_pteta_high_cb",_CommonSpect,_JetVars+_BTagVars+_SIPVars+_IpVars+_MvaIdVars+_InnerHitsVars) ),
            ( lambda x: x.pt >  10 and abs(x.eta) >= 0.8 and abs(x.eta) <  1.479 , MVATool("BDTG",basepath%"el_pteta_high_fb",_CommonSpect,_JetVars+_BTagVars+_SIPVars+_IpVars+_MvaIdVars+_InnerHitsVars) ),
            ( lambda x: x.pt >  10 and abs(x.eta) >= 1.479                       , MVATool("BDTG",basepath%"el_pteta_high_ec",_CommonSpect,_JetVars+_BTagVars+_SIPVars+_IpVars+_MvaIdVars+_InnerHitsVars) ),
        ])
    def __call__(self,lep,ncorr=0):
        if   abs(lep.pdgId) == 11: return self.el(lep,ncorr)
        elif abs(lep.pdgId) == 13: return self.mu(lep,ncorr)
        else: return -99


class LeptonMVANoIp:
    def __init__(self,basepath):
        global _CommonVars, _CommonSpect, _ElectronVars
        self.mu = CategorizedMVA([
            ( lambda x: x.pt <= 15 and abs(x.eta) <  1.5 , MVATool("BDTG",basepath%"mu_pteta_low_b", _CommonSpect,_IsoVars+_JetVars+_BTagVars) ),
            ( lambda x: x.pt <= 15 and abs(x.eta) >= 1.5 , MVATool("BDTG",basepath%"mu_pteta_low_e", _CommonSpect,_IsoVars+_JetVars+_BTagVars) ),
            ( lambda x: x.pt >  15 and abs(x.eta) <  1.5 , MVATool("BDTG",basepath%"mu_pteta_high_b",_CommonSpect,_IsoVars+_JetVars+_BTagVars) ),
            ( lambda x: x.pt >  15 and abs(x.eta) >= 1.5 , MVATool("BDTG",basepath%"mu_pteta_high_e",_CommonSpect,_IsoVars+_JetVars+_BTagVars) ),
        ])
        self.el = CategorizedMVA([
            ( lambda x: x.pt <= 10 and abs(x.eta) <  0.8                         , MVATool("BDTG",basepath%"el_pteta_low_cb", _CommonSpect,_IsoVars+_JetVars+_BTagVars+_MvaIdVars+_InnerHitsVars) ),
            ( lambda x: x.pt <= 10 and abs(x.eta) >= 0.8 and abs(x.eta) <  1.479 , MVATool("BDTG",basepath%"el_pteta_low_fb", _CommonSpect,_IsoVars+_JetVars+_BTagVars+_MvaIdVars+_InnerHitsVars) ),
            ( lambda x: x.pt <= 10 and abs(x.eta) >= 1.479                       , MVATool("BDTG",basepath%"el_pteta_low_ec", _CommonSpect,_IsoVars+_JetVars+_BTagVars+_MvaIdVars+_InnerHitsVars) ),
            ( lambda x: x.pt >  10 and abs(x.eta) <  0.8                         , MVATool("BDTG",basepath%"el_pteta_high_cb",_CommonSpect,_IsoVars+_JetVars+_BTagVars+_MvaIdVars+_InnerHitsVars) ),
            ( lambda x: x.pt >  10 and abs(x.eta) >= 0.8 and abs(x.eta) <  1.479 , MVATool("BDTG",basepath%"el_pteta_high_fb",_CommonSpect,_IsoVars+_JetVars+_BTagVars+_MvaIdVars+_InnerHitsVars) ),
            ( lambda x: x.pt >  10 and abs(x.eta) >= 1.479                       , MVATool("BDTG",basepath%"el_pteta_high_ec",_CommonSpect,_IsoVars+_JetVars+_BTagVars+_MvaIdVars+_InnerHitsVars) ),
        ])
    def __call__(self,lep,ncorr=0):
        if   abs(lep.pdgId) == 11: return self.el(lep,ncorr)
        elif abs(lep.pdgId) == 13: return self.mu(lep,ncorr)
        else: return -99



class LeptonMVANoJet:
    def __init__(self,basepath):
        global _CommonVars, _CommonSpect, _ElectronVars
        self.mu = CategorizedMVA([
            ( lambda x: x.pt <= 15 and abs(x.eta) <  1.5 , MVATool("BDTG",basepath%"mu_pteta_low_b", _CommonSpect,_IsoVars+_BTagVars+_SIPVars+_IpVars) ),
            ( lambda x: x.pt <= 15 and abs(x.eta) >= 1.5 , MVATool("BDTG",basepath%"mu_pteta_low_e", _CommonSpect,_IsoVars+_BTagVars+_SIPVars+_IpVars) ),
            ( lambda x: x.pt >  15 and abs(x.eta) <  1.5 , MVATool("BDTG",basepath%"mu_pteta_high_b",_CommonSpect,_IsoVars+_BTagVars+_SIPVars+_IpVars) ),
            ( lambda x: x.pt >  15 and abs(x.eta) >= 1.5 , MVATool("BDTG",basepath%"mu_pteta_high_e",_CommonSpect,_IsoVars+_BTagVars+_SIPVars+_IpVars) ),
        ])
        self.el = CategorizedMVA([
            ( lambda x: x.pt <= 10 and abs(x.eta) <  0.8                         , MVATool("BDTG",basepath%"el_pteta_low_cb", _CommonSpect,_IsoVars+_BTagVars+_SIPVars+_IpVars+_MvaIdVars+_InnerHitsVars) ),
            ( lambda x: x.pt <= 10 and abs(x.eta) >= 0.8 and abs(x.eta) <  1.479 , MVATool("BDTG",basepath%"el_pteta_low_fb", _CommonSpect,_IsoVars+_BTagVars+_SIPVars+_IpVars+_MvaIdVars+_InnerHitsVars) ),
            ( lambda x: x.pt <= 10 and abs(x.eta) >= 1.479                       , MVATool("BDTG",basepath%"el_pteta_low_ec", _CommonSpect,_IsoVars+_BTagVars+_SIPVars+_IpVars+_MvaIdVars+_InnerHitsVars) ),
            ( lambda x: x.pt >  10 and abs(x.eta) <  0.8                         , MVATool("BDTG",basepath%"el_pteta_high_cb",_CommonSpect,_IsoVars+_BTagVars+_SIPVars+_IpVars+_MvaIdVars+_InnerHitsVars) ),
            ( lambda x: x.pt >  10 and abs(x.eta) >= 0.8 and abs(x.eta) <  1.479 , MVATool("BDTG",basepath%"el_pteta_high_fb",_CommonSpect,_IsoVars+_BTagVars+_SIPVars+_IpVars+_MvaIdVars+_InnerHitsVars) ),
            ( lambda x: x.pt >  10 and abs(x.eta) >= 1.479                       , MVATool("BDTG",basepath%"el_pteta_high_ec",_CommonSpect,_IsoVars+_BTagVars+_SIPVars+_IpVars+_MvaIdVars+_InnerHitsVars) ),
        ])
    def __call__(self,lep,ncorr=0):
        if   abs(lep.pdgId) == 11: return self.el(lep,ncorr)
        elif abs(lep.pdgId) == 13: return self.mu(lep,ncorr)
        else: return -99


class LeptonMVANoMvaId:
    def __init__(self,basepath):
        global _CommonVars, _CommonSpect, _ElectronVars
        self.mu = CategorizedMVA([
            ( lambda x: x.pt <= 15 and abs(x.eta) <  1.5 , MVATool("BDTG",basepath%"mu_pteta_low_b", _CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_IpVars) ),
            ( lambda x: x.pt <= 15 and abs(x.eta) >= 1.5 , MVATool("BDTG",basepath%"mu_pteta_low_e", _CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_IpVars) ),
            ( lambda x: x.pt >  15 and abs(x.eta) <  1.5 , MVATool("BDTG",basepath%"mu_pteta_high_b",_CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_IpVars) ),
            ( lambda x: x.pt >  15 and abs(x.eta) >= 1.5 , MVATool("BDTG",basepath%"mu_pteta_high_e",_CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_IpVars) ),
        ])
        self.el = CategorizedMVA([
            ( lambda x: x.pt <= 10 and abs(x.eta) <  0.8                         , MVATool("BDTG",basepath%"el_pteta_low_cb", _CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_IpVars+_InnerHitsVars) ),
            ( lambda x: x.pt <= 10 and abs(x.eta) >= 0.8 and abs(x.eta) <  1.479 , MVATool("BDTG",basepath%"el_pteta_low_fb", _CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_IpVars+_InnerHitsVars) ),
            ( lambda x: x.pt <= 10 and abs(x.eta) >= 1.479                       , MVATool("BDTG",basepath%"el_pteta_low_ec", _CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_IpVars+_InnerHitsVars) ),
            ( lambda x: x.pt >  10 and abs(x.eta) <  0.8                         , MVATool("BDTG",basepath%"el_pteta_high_cb",_CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_IpVars+_InnerHitsVars) ),
            ( lambda x: x.pt >  10 and abs(x.eta) >= 0.8 and abs(x.eta) <  1.479 , MVATool("BDTG",basepath%"el_pteta_high_fb",_CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_IpVars+_InnerHitsVars) ),
            ( lambda x: x.pt >  10 and abs(x.eta) >= 1.479                       , MVATool("BDTG",basepath%"el_pteta_high_ec",_CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_IpVars+_InnerHitsVars) ),
        ])
    def __call__(self,lep,ncorr=0):
        if   abs(lep.pdgId) == 11: return self.el(lep,ncorr)
        elif abs(lep.pdgId) == 13: return self.mu(lep,ncorr)
        else: return -99


class LeptonMVANoBtag:
    def __init__(self,basepath):
        global _CommonVars, _CommonSpect, _ElectronVars
        self.mu = CategorizedMVA([
            ( lambda x: x.pt <= 15 and abs(x.eta) <  1.5 , MVATool("BDTG",basepath%"mu_pteta_low_b", _CommonSpect,_IsoVars+_JetVars+_SIPVars+_IpVars) ),
            ( lambda x: x.pt <= 15 and abs(x.eta) >= 1.5 , MVATool("BDTG",basepath%"mu_pteta_low_e", _CommonSpect,_IsoVars+_JetVars+_SIPVars+_IpVars) ),
            ( lambda x: x.pt >  15 and abs(x.eta) <  1.5 , MVATool("BDTG",basepath%"mu_pteta_high_b",_CommonSpect,_IsoVars+_JetVars+_SIPVars+_IpVars) ),
            ( lambda x: x.pt >  15 and abs(x.eta) >= 1.5 , MVATool("BDTG",basepath%"mu_pteta_high_e",_CommonSpect,_IsoVars+_JetVars+_SIPVars+_IpVars) ),
        ])
        self.el = CategorizedMVA([
            ( lambda x: x.pt <= 10 and abs(x.eta) <  0.8                         , MVATool("BDTG",basepath%"el_pteta_low_cb", _CommonSpect,_IsoVars+_JetVars+_SIPVars+_IpVars+_MvaIdVars+_InnerHitsVars) ),
            ( lambda x: x.pt <= 10 and abs(x.eta) >= 0.8 and abs(x.eta) <  1.479 , MVATool("BDTG",basepath%"el_pteta_low_fb", _CommonSpect,_IsoVars+_JetVars+_SIPVars+_IpVars+_MvaIdVars+_InnerHitsVars) ),
            ( lambda x: x.pt <= 10 and abs(x.eta) >= 1.479                       , MVATool("BDTG",basepath%"el_pteta_low_ec", _CommonSpect,_IsoVars+_JetVars+_SIPVars+_IpVars+_MvaIdVars+_InnerHitsVars) ),
            ( lambda x: x.pt >  10 and abs(x.eta) <  0.8                         , MVATool("BDTG",basepath%"el_pteta_high_cb",_CommonSpect,_IsoVars+_JetVars+_SIPVars+_IpVars+_MvaIdVars+_InnerHitsVars) ),
            ( lambda x: x.pt >  10 and abs(x.eta) >= 0.8 and abs(x.eta) <  1.479 , MVATool("BDTG",basepath%"el_pteta_high_fb",_CommonSpect,_IsoVars+_JetVars+_SIPVars+_IpVars+_MvaIdVars+_InnerHitsVars) ),
            ( lambda x: x.pt >  10 and abs(x.eta) >= 1.479                       , MVATool("BDTG",basepath%"el_pteta_high_ec",_CommonSpect,_IsoVars+_JetVars+_SIPVars+_IpVars+_MvaIdVars+_InnerHitsVars) ),
        ])
    def __call__(self,lep,ncorr=0):
        if   abs(lep.pdgId) == 11: return self.el(lep,ncorr)
        elif abs(lep.pdgId) == 13: return self.mu(lep,ncorr)
        else: return -99


class LeptonMVANoInnerHits:
    def __init__(self,basepath):
        global _CommonVars, _CommonSpect, _ElectronVars
        self.mu = CategorizedMVA([
            ( lambda x: x.pt <= 15 and abs(x.eta) <  1.5 , MVATool("BDTG",basepath%"mu_pteta_low_b", _CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_IpVars) ),
            ( lambda x: x.pt <= 15 and abs(x.eta) >= 1.5 , MVATool("BDTG",basepath%"mu_pteta_low_e", _CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_IpVars) ),
            ( lambda x: x.pt >  15 and abs(x.eta) <  1.5 , MVATool("BDTG",basepath%"mu_pteta_high_b",_CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_IpVars) ),
            ( lambda x: x.pt >  15 and abs(x.eta) >= 1.5 , MVATool("BDTG",basepath%"mu_pteta_high_e",_CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_IpVars) ),
        ])
        self.el = CategorizedMVA([
            ( lambda x: x.pt <= 10 and abs(x.eta) <  0.8                         , MVATool("BDTG",basepath%"el_pteta_low_cb", _CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_IpVars+_MvaIdVars) ),
            ( lambda x: x.pt <= 10 and abs(x.eta) >= 0.8 and abs(x.eta) <  1.479 , MVATool("BDTG",basepath%"el_pteta_low_fb", _CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_IpVars+_MvaIdVars) ),
            ( lambda x: x.pt <= 10 and abs(x.eta) >= 1.479                       , MVATool("BDTG",basepath%"el_pteta_low_ec", _CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_IpVars+_MvaIdVars) ),
            ( lambda x: x.pt >  10 and abs(x.eta) <  0.8                         , MVATool("BDTG",basepath%"el_pteta_high_cb",_CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_IpVars+_MvaIdVars) ),
            ( lambda x: x.pt >  10 and abs(x.eta) >= 0.8 and abs(x.eta) <  1.479 , MVATool("BDTG",basepath%"el_pteta_high_fb",_CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_IpVars+_MvaIdVars) ),
            ( lambda x: x.pt >  10 and abs(x.eta) >= 1.479                       , MVATool("BDTG",basepath%"el_pteta_high_ec",_CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_IpVars+_MvaIdVars) ),
        ])
    def __call__(self,lep,ncorr=0):
        if   abs(lep.pdgId) == 11: return self.el(lep,ncorr)
        elif abs(lep.pdgId) == 13: return self.mu(lep,ncorr)
        else: return -99



class LeptonMVANoSIP:
    def __init__(self,basepath):
        global _CommonVars, _CommonSpect, _ElectronVars
        self.mu = CategorizedMVA([
            ( lambda x: x.pt <= 15 and abs(x.eta) <  1.5 , MVATool("BDTG",basepath%"mu_pteta_low_b", _CommonSpect,_IsoVars+_JetVars+_BTagVars+_IpVars) ),
            ( lambda x: x.pt <= 15 and abs(x.eta) >= 1.5 , MVATool("BDTG",basepath%"mu_pteta_low_e", _CommonSpect,_IsoVars+_JetVars+_BTagVars+_IpVars) ),
            ( lambda x: x.pt >  15 and abs(x.eta) <  1.5 , MVATool("BDTG",basepath%"mu_pteta_high_b",_CommonSpect,_IsoVars+_JetVars+_BTagVars+_IpVars) ),
            ( lambda x: x.pt >  15 and abs(x.eta) >= 1.5 , MVATool("BDTG",basepath%"mu_pteta_high_e",_CommonSpect,_IsoVars+_JetVars+_BTagVars+_IpVars) ),
        ])
        self.el = CategorizedMVA([
            ( lambda x: x.pt <= 10 and abs(x.eta) <  0.8                         , MVATool("BDTG",basepath%"el_pteta_low_cb", _CommonSpect,_IsoVars+_JetVars+_BTagVars+_IpVars+_MvaIdVars+_InnerHitsVars) ),
            ( lambda x: x.pt <= 10 and abs(x.eta) >= 0.8 and abs(x.eta) <  1.479 , MVATool("BDTG",basepath%"el_pteta_low_fb", _CommonSpect,_IsoVars+_JetVars+_BTagVars+_IpVars+_MvaIdVars+_InnerHitsVars) ),
            ( lambda x: x.pt <= 10 and abs(x.eta) >= 1.479                       , MVATool("BDTG",basepath%"el_pteta_low_ec", _CommonSpect,_IsoVars+_JetVars+_BTagVars+_IpVars+_MvaIdVars+_InnerHitsVars) ),
            ( lambda x: x.pt >  10 and abs(x.eta) <  0.8                         , MVATool("BDTG",basepath%"el_pteta_high_cb",_CommonSpect,_IsoVars+_JetVars+_BTagVars+_IpVars+_MvaIdVars+_InnerHitsVars) ),
            ( lambda x: x.pt >  10 and abs(x.eta) >= 0.8 and abs(x.eta) <  1.479 , MVATool("BDTG",basepath%"el_pteta_high_fb",_CommonSpect,_IsoVars+_JetVars+_BTagVars+_IpVars+_MvaIdVars+_InnerHitsVars) ),
            ( lambda x: x.pt >  10 and abs(x.eta) >= 1.479                       , MVATool("BDTG",basepath%"el_pteta_high_ec",_CommonSpect,_IsoVars+_JetVars+_BTagVars+_IpVars+_MvaIdVars+_InnerHitsVars) ),
        ])
    def __call__(self,lep,ncorr=0):
        if   abs(lep.pdgId) == 11: return self.el(lep,ncorr)
        elif abs(lep.pdgId) == 13: return self.mu(lep,ncorr)
        else: return -99



class LeptonMVANodxydz:
    def __init__(self,basepath):
        global _CommonVars, _CommonSpect, _ElectronVars
        self.mu = CategorizedMVA([
            ( lambda x: x.pt <= 15 and abs(x.eta) <  1.5 , MVATool("BDTG",basepath%"mu_pteta_low_b", _CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars) ),
            ( lambda x: x.pt <= 15 and abs(x.eta) >= 1.5 , MVATool("BDTG",basepath%"mu_pteta_low_e", _CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars) ),
            ( lambda x: x.pt >  15 and abs(x.eta) <  1.5 , MVATool("BDTG",basepath%"mu_pteta_high_b",_CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars) ),
            ( lambda x: x.pt >  15 and abs(x.eta) >= 1.5 , MVATool("BDTG",basepath%"mu_pteta_high_e",_CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars) ),
        ])
        self.el = CategorizedMVA([
            ( lambda x: x.pt <= 10 and abs(x.eta) <  0.8                         , MVATool("BDTG",basepath%"el_pteta_low_cb", _CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_MvaIdVars+_InnerHitsVars) ),
            ( lambda x: x.pt <= 10 and abs(x.eta) >= 0.8 and abs(x.eta) <  1.479 , MVATool("BDTG",basepath%"el_pteta_low_fb", _CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_MvaIdVars+_InnerHitsVars) ),
            ( lambda x: x.pt <= 10 and abs(x.eta) >= 1.479                       , MVATool("BDTG",basepath%"el_pteta_low_ec", _CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_MvaIdVars+_InnerHitsVars) ),
            ( lambda x: x.pt >  10 and abs(x.eta) <  0.8                         , MVATool("BDTG",basepath%"el_pteta_high_cb",_CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_MvaIdVars+_InnerHitsVars) ),
            ( lambda x: x.pt >  10 and abs(x.eta) >= 0.8 and abs(x.eta) <  1.479 , MVATool("BDTG",basepath%"el_pteta_high_fb",_CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_MvaIdVars+_InnerHitsVars) ),
            ( lambda x: x.pt >  10 and abs(x.eta) >= 1.479                       , MVATool("BDTG",basepath%"el_pteta_high_ec",_CommonSpect,_IsoVars+_JetVars+_BTagVars+_SIPVars+_MvaIdVars+_InnerHitsVars) ),
        ])
    def __call__(self,lep,ncorr=0):
        if   abs(lep.pdgId) == 11: return self.el(lep,ncorr)
        elif abs(lep.pdgId) == 13: return self.mu(lep,ncorr)
        else: return -99




        

class LepMVATreeProducer(Module):
    def __init__(self,name,booker,path,data,fast=False,others=False):
        Module.__init__(self,name,booker)
        self.mva = LeptonMVA(path+"/weights/%s_BDTG.weights.xml")
        self.others = others
        if self.others:
            self.mvaNoSIP = LeptonMVANoSIP(path+"/weightsNoSIP/%s_BDTG.weights.xml")
            self.mvaNodxydz = LeptonMVANodxydz(path+"/weightsNodxydz/%s_BDTG.weights.xml")
            #self.mvaNoIso = LeptonMVANoIso(path+"/weightsNoIso/%s_BDTG.weights.xml")
            #self.mvaNoIp = LeptonMVANoIp(path+"/weightsNoIp/%s_BDTG.weights.xml")
            #self.mvaNoJet = LeptonMVANoJet(path+"/weightsNoJet/%s_BDTG.weights.xml")
            #self.mvaNoMvaId = LeptonMVANoMvaId(path+"/weightsNoMvaId/%s_BDTG.weights.xml")
            #self.mvaNoBtag = LeptonMVANoBtag(path+"/weightsNoBtag/%s_BDTG.weights.xml")
            #self.mvaNoInnerHits = LeptonMVANoInnerHits(path+"/weightsNoInnerHits/%s_BDTG.weights.xml")       
        self.data = data
        self.fast = fast
    def beginJob(self):
        self.t = PyTree(self.book("TTree","t","t"))
        for i in range(8):
            self.t.branch("LepGood%d_mvaNew" % (i+1),"F")
            if self.others:
                self.t.branch("LepGood%d_mvaNoSIP" % (i+1),"F")
                self.t.branch("LepGood%d_mvaNodxydz" % (i+1),"F")
                #self.t.branch("LepGood%d_mvaNoIso" % (i+1),"F")
                #self.t.branch("LepGood%d_mvaNoIp" % (i+1),"F")
                #self.t.branch("LepGood%d_mvaNoJet" % (i+1),"F")                        
                #self.t.branch("LepGood%d_mvaNoMvaId" % (i+1),"F")
                #self.t.branch("LepGood%d_mvaNoBtag" % (i+1),"F")
                #self.t.branch("LepGood%d_mvaNoInnerHits" % (i+1),"F")
            if not self.data and not self.fast:
                self.t.branch("LepGood%d_mvaNewUncorr"     % (i+1),"F")
                self.t.branch("LepGood%d_mvaNewDoubleCorr" % (i+1),"F")
    def analyze(self,event):
        lep = Collection(event,"LepGood","nLepGood",8)
        for i,l in enumerate(lep):
            if self.data:
                setattr(self.t, "LepGood%d_mvaNew" % (i+1), self.mva(l,ncorr=0))
                if self.others:
                    setattr(self.t, "LepGood%d_mvaNoSIP" % (i+1), self.mvaNoSIP(l,ncorr=0))
                    setattr(self.t, "LepGood%d_mvaNodxydz" % (i+1), self.mvaNodxydz(l,ncorr=0))
                    #setattr(self.t, "LepGood%d_mvaNoIso" % (i+1), self.mvaNoIso(l,ncorr=0))
                    #setattr(self.t, "LepGood%d_mvaNoIp" % (i+1), self.mvaNoIp(l,ncorr=0))
                    #setattr(self.t, "LepGood%d_mvaNoJet" % (i+1), self.mvaNoJet(l,ncorr=0))
                    #setattr(self.t, "LepGood%d_mvaNoMvaId" % (i+1), self.mvaNoMvaId(l,ncorr=0))
                    #setattr(self.t, "LepGood%d_mvaNoBtag" % (i+1), self.mvaNoBtag(l,ncorr=0))
                    #setattr(self.t, "LepGood%d_mvaNoInnerHits" % (i+1), self.mvaNoInnerHits(l,ncorr=0))
            else: 
                setattr(self.t, "LepGood%d_mvaNew" % (i+1), self.mva(l,ncorr=1))
                if self.others:
                    setattr(self.t, "LepGood%d_mvaNoSIP" % (i+1), self.mvaNoSIP(l,ncorr=1))
                    setattr(self.t, "LepGood%d_mvaNodxydz" % (i+1), self.mvaNodxydz(l,ncorr=1))
                    #setattr(self.t, "LepGood%d_mvaNoIso" % (i+1), self.mvaNoIso(l,ncorr=1))
                    #setattr(self.t, "LepGood%d_mvaNoIp" % (i+1), self.mvaNoIp(l,ncorr=1))
                    #setattr(self.t, "LepGood%d_mvaNoJet" % (i+1), self.mvaNoJet(l,ncorr=1))
                    #setattr(self.t, "LepGood%d_mvaNoMvaId" % (i+1), self.mvaNoMvaId(l,ncorr=1))
                    #setattr(self.t, "LepGood%d_mvaNoBtag" % (i+1), self.mvaNoBtag(l,ncorr=1))
                    #setattr(self.t, "LepGood%d_mvaNoInnerHits" % (i+1), self.mvaNoInnerHits(l,ncorr=1))
                if not self.fast:
                    setattr(self.t, "LepGood%d_mvaNewUncorr"     % (i+1), self.mva(l,ncorr=0))
                    setattr(self.t, "LepGood%d_mvaNewDoubleCorr" % (i+1), self.mva(l,ncorr=2))
        for i in xrange(len(lep),8):
            setattr(self.t, "LepGood%d_mvaNew" % (i+1), -99.)
            if self.others:
                setattr(self.t, "LepGood%d_mvaNoSIP" % (i+1), -99.)
                setattr(self.t, "LepGood%d_mvaNodxydz" % (i+1), -99.)
                #setattr(self.t, "LepGood%d_mvaNoIso" % (i+1), -99.)
                #setattr(self.t, "LepGood%d_mvaNoIp" % (i+1), -99.)
                #setattr(self.t, "LepGood%d_mvaNoJet" % (i+1), -99.)
                #setattr(self.t, "LepGood%d_mvaNoMvaId" % (i+1), -99.)
                #setattr(self.t, "LepGood%d_mvaNoBtag" % (i+1), -99.)
                #setattr(self.t, "LepGood%d_mvaNoInnerHits" % (i+1), -99.)
            if not self.data and not self.fast:
                setattr(self.t, "LepGood%d_mvaNewUncorr"     % (i+1), -99.)
                setattr(self.t, "LepGood%d_mvaNewDoubleCorr" % (i+1), -99.)
        self.t.fill()

import os, itertools

from optparse import OptionParser
parser = OptionParser(usage="%prog [options] <TREE_DIR> <TRAINING>")
parser.add_option("-d", "--dataset", dest="datasets",  type="string", default=[], action="append", help="Process only this dataset (or dataset if specified multiple times)");
parser.add_option("-c", "--chunk",   dest="chunks",    type="int",    default=[], action="append", help="Process only these chunks (works only if a single dataset is selected with -d)");
parser.add_option("-N", "--events",  dest="chunkSize", type="int",    default=500000, help="Default chunk size when splitting trees");
parser.add_option("-j", "--jobs",    dest="jobs",      type="int",    default=1, help="Use N threads");
parser.add_option("-a", "--all",     dest="allMVAs",   action="store_true", default=False, help="Run also all the other special trainings, not just the main one");
parser.add_option("-p", "--pretend", dest="pretend",   action="store_true", default=False, help="Don't run anything");
parser.add_option("-q", "--queue",   dest="queue",     type="string", default=None, help="Run jobs on lxbatch instead of locally");
(options, args) = parser.parse_args()

if len(args) != 2: 
    print "Usage: program <TREE_DIR> <TRAINING>"
    exit()
if len(options.chunks) != 0 and len(options.datasets) != 1:
    print "must specify a single dataset with -d if using -c to select chunks"
    exit()

jobs = []
for D in glob(args[0]+"/*"):
    fname = D+"/ttHLepTreeProducerBase/ttHLepTreeProducerBase_tree.root"
    if os.path.exists(fname):
        short = os.path.basename(D)
        if options.datasets != []:
            if short not in options.datasets: continue
        data = ("DoubleMu" in short or "MuEG" in short or "DoubleElectron" in short or "SingleMu" in short)
        f = ROOT.TFile.Open(fname);
        t = f.Get("ttHLepTreeProducerBase")
        entries = t.GetEntries()
        f.Close()
        chunk = options.chunkSize
        if entries < chunk:
            print "  ",os.path.basename(D),("  DATA" if data else "  MC")," single chunk"
            jobs.append((short,fname,"%s/lepMVAFriend_%s.root" % (args[1],short),data,xrange(entries),-1))
        else:
            nchunk = int(ceil(entries/float(chunk)))
            print "  ",os.path.basename(D),("  DATA" if data else "  MC")," %d chunks" % nchunk
            for i in xrange(nchunk):
                if options.chunks != []:
                    if i not in options.chunks: continue
                r = xrange(int(i*chunk),min(int((i+1)*chunk),entries))
                jobs.append((short,fname,"%s/lepMVAFriend_%s.chunk%d.root" % (args[1],short,i),data,r,i))
print "\n"
print "I have %d taks to process" % len(jobs)

if options.queue:
    import os, sys
    basecmd = "bsub -q {queue} {dir}/lxbatch_runner.sh {dir} {cmssw} python {self} -N {chunkSize} {data} {training}".format(
                queue = options.queue, dir = os.getcwd(), cmssw = os.environ['CMSSW_BASE'], 
                self=sys.argv[0], chunkSize=options.chunkSize, data=args[0], training=args[1]
            )
    ## forward additional options if needed
    if options.allMVAs: basecmd += " --all";
    # specify what to do
    for (name,fin,fout,data,range,chunk) in jobs:
        if chunk != -1:
            print "{base} -d {data} -c {chunk}".format(base=basecmd, data=name, chunk=chunk)
        else:
            print "{base} -d {data}".format(base=basecmd, data=name, chunk=chunk)
    exit()

maintimer = ROOT.TStopwatch()
def _runIt(myargs):
    (name,fin,fout,data,range,chunk) = myargs
    timer = ROOT.TStopwatch()
    fb = ROOT.TFile(fin)
    tb = fb.Get("ttHLepTreeProducerBase")
    nev = tb.GetEntries()
    if options.pretend:
        print "==== pretending to run %s (%d entries, %s) ====" % (name, nev, fout)
        return (name,(nev,0))
    print "==== %s starting (%d entries) ====" % (name, nev)
    booker = Booker(fout)
    el = EventLoop([ LepMVATreeProducer("newMVA",booker,args[1],data,fast=True,others=options.allMVAs), ])
    el.loop([tb], eventRange=range)
    booker.done()
    fb.Close()
    time = timer.RealTime()
    print "=== %s done (%d entries, %.0f s, %.0f e/s) ====" % ( name, nev, time,(nev/time) )
    return (name,(nev,time))

from multiprocessing import Pool
pool = Pool(options.jobs)
ret  = dict(pool.map(_runIt, jobs))
fulltime = maintimer.RealTime()
totev   = sum([ev   for (ev,time) in ret.itervalues()])
tottime = sum([time for (ev,time) in ret.itervalues()])
print "Done %d tasks in %.1f min (%d entries, %.1f min)" % (len(jobs),fulltime/60.,totev,tottime/60.)

