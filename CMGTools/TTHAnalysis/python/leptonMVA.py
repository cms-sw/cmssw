from array import array
from math import *
from PhysicsTools.HeppyCore.utils.deltar import deltaR
from CMGTools.TTHAnalysis.analyzers.ntupleTypes import ptRelv1

import ROOT
#import os
#if "/smearer_cc.so" not in ROOT.gSystem.GetLibraries(): 
#    ROOT.gROOT.ProcessLine(".L %s/src/CMGTools/TTHAnalysis/python/plotter/smearer.cc+" % os.environ['CMSSW_BASE']);
#if "/mcCorrections_cc.so" not in ROOT.gSystem.GetLibraries(): 
#    ROOT.gROOT.ProcessLine(".L %s/src/CMGTools/TTHAnalysis/python/plotter/mcCorrections.cc+" % os.environ['CMSSW_BASE']);


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
                self.var[0] = self.corrfunc(self.var[0], lep.pdgId(),lep.pt(),lep.eta(),lep.mcMatchId,lep.mcMatchAny)
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
_CommonVars = {
 'Susy':[ 

    MVAVar("LepGood_neuRelIso03 := LepGood_relIso03 - LepGood_chargedHadRelIso03",lambda x: x.relIso03 - x.chargedHadronIsoR(0.3)/x.pt()),  
    MVAVar("LepGood_chRelIso03 := LepGood_chargedHadRelIso03",lambda x: x.chargedHadronIsoR(0.3)/x.pt()),
    MVAVar("LepGood_jetDR := min(LepGood_jetDR,0.5)", lambda x : min(deltaR(x.eta(),x.phi(),x.jet.eta(),x.jet.phi()),0.5)), #, corrfunc=ROOT.correctJetDRMC),
    MVAVar("LepGood_jetPtRatio := min(LepGood_jetPtRatio,1.5)", lambda x : min(x.pt()/x.jet.pt(),1.5)), #, corrfunc=ROOT.correctJetPtRatioMC),
    MVAVar("LepGood_jetBTagCSV := max(LepGood_jetBTagCSV,0)", lambda x : max( (x.jet.btag('pfCombinedInclusiveSecondaryVertexV2BJetTags') if hasattr(x.jet, 'btag') else -99) ,0.)),
    MVAVar("LepGood_sip3d",lambda x: x.sip3D()), #, corrfunc=ROOT.scaleSip3dMC),
    MVAVar("LepGood_dxy := log(abs(LepGood_dxy))",lambda x: log(abs(x.dxy()))), #, corrfunc=ROOT.scaleDxyMC),
    MVAVar("LepGood_dz  := log(abs(LepGood_dz))", lambda x: log(abs(x.dz()))), #, corrfunc=ROOT.scaleDzMC),
 ],

'SusyWithBoost':[ 
    MVAVar("LepGood_miniRelIsoCharged",lambda x: getattr(x,'miniAbsIsoCharged',-99)/x.pt()), 
    MVAVar("LepGood_miniRelIsoNeutral",lambda x: getattr(x,'miniAbsIsoNeutral',-99)/x.pt()), 
    MVAVar("LepGood_jetPtRelV1",lambda x: ptRelv1(x.p4(),x.jet.p4())), 
    MVAVar("LepGood_neuRelIso03 := LepGood_relIso03 - LepGood_chargedHadRelIso03",lambda x: x.relIso03 - x.chargedHadronIsoR(0.3)/x.pt()),  
    MVAVar("LepGood_chRelIso03 := LepGood_chargedHadRelIso03",lambda x: x.chargedHadronIsoR(0.3)/x.pt()),
    MVAVar("LepGood_jetDR := min(LepGood_jetDR,0.5)", lambda x : min(deltaR(x.eta(),x.phi(),x.jet.eta(),x.jet.phi()),0.5)), #, corrfunc=ROOT.correctJetDRMC),
    MVAVar("LepGood_jetPtRatio := min(LepGood_jetPtRatio,1.5)", lambda x : min(x.pt()/x.jet.pt(),1.5)), #, corrfunc=ROOT.correctJetPtRatioMC),
    MVAVar("LepGood_jetBTagCSV := max(LepGood_jetBTagCSV,0)", lambda x : max( (x.jet.btag('pfCombinedInclusiveSecondaryVertexV2BJetTags') if hasattr(x.jet, 'btag') else -99) ,0.)),
    MVAVar("LepGood_sip3d",lambda x: x.sip3D()), #, corrfunc=ROOT.scaleSip3dMC),
    MVAVar("LepGood_dxy := log(abs(LepGood_dxy))",lambda x: log(abs(x.dxy()))), #, corrfunc=ROOT.scaleDxyMC),
    MVAVar("LepGood_dz  := log(abs(LepGood_dz))", lambda x: log(abs(x.dz()))), #, corrfunc=ROOT.scaleDzMC),
]

}

_MuonVars = {
 'Susy': [
    MVAVar("LepGood_segmentCompatibility",lambda x: x.segmentCompatibility()), 
 ],

 'SusyWithBoost': [
    MVAVar("LepGood_segmentCompatibility",lambda x: x.segmentCompatibility()), 
 ]

}

_ElectronVars = {
 'Susy': [
    MVAVar("LepGood_mvaIdPhys14",lambda x: x.mvaRun2("NonTrigPhys14")),    
 ],
 
 'SusyWithBoost': [
    MVAVar("LepGood_mvaIdPhys14",lambda x: x.mvaRun2("NonTrigPhys14")),    
 ]

}


class LeptonMVA:
    def __init__(self, kind, basepath, isMC):
        global _CommonVars, _CommonSpect, _ElectronVars
        #print "Creating LeptonMVA of kind %s, base path %s" % (kind, basepath)
        self._isMC = isMC
        self._kind = kind
        muVars = _CommonVars[kind] + _MuonVars[kind]
        elVars = _CommonVars[kind] + _ElectronVars[kind]
        self.mu = CategorizedMVA([
            ( lambda x: x.pt() <= 10, MVATool("BDTG",basepath%"mu_pteta_low", _CommonSpect,muVars) ),
            ( lambda x: x.pt() > 10 and x.pt() <= 25 and abs(x.eta()) <  1.5 , MVATool("BDTG",basepath%"mu_pteta_medium_b", _CommonSpect,muVars) ),
            ( lambda x: x.pt() > 10 and x.pt() <= 25 and abs(x.eta()) >= 1.5 , MVATool("BDTG",basepath%"mu_pteta_medium_e", _CommonSpect,muVars) ),
            ( lambda x: x.pt() >  25 and abs(x.eta()) <  1.5 , MVATool("BDTG",basepath%"mu_pteta_high_b",_CommonSpect,muVars) ),
            ( lambda x: x.pt() >  25 and abs(x.eta()) >= 1.5 , MVATool("BDTG",basepath%"mu_pteta_high_e",_CommonSpect,muVars) ),
        ])
        self.el = CategorizedMVA([
            ( lambda x: x.pt() <= 10                          , MVATool("BDTG",basepath%"el_pteta_low", _CommonSpect,elVars) ),    
            ( lambda x: x.pt() >  10 and x.pt() <= 25 and abs(x.eta()) <  0.8                         , MVATool("BDTG",basepath%"el_pteta_medium_cb", _CommonSpect,elVars) ),
            ( lambda x: x.pt() >  10 and x.pt() <= 25 and abs(x.eta()) >= 0.8 and abs(x.eta()) <  1.479 , MVATool("BDTG",basepath%"el_pteta_medium_fb", _CommonSpect,elVars) ),
            ( lambda x: x.pt() >  10 and x.pt() <= 25 and abs(x.eta()) >= 1.479                       , MVATool("BDTG",basepath%"el_pteta_medium_ec", _CommonSpect,elVars) ),
            ( lambda x: x.pt() >  25 and abs(x.eta()) <  0.8                         , MVATool("BDTG",basepath%"el_pteta_high_cb",_CommonSpect,elVars) ),
            ( lambda x: x.pt() >  25 and abs(x.eta()) >= 0.8 and abs(x.eta()) <  1.479 , MVATool("BDTG",basepath%"el_pteta_high_fb",_CommonSpect,elVars) ),
            ( lambda x: x.pt() >  25 and abs(x.eta()) >= 1.479                       , MVATool("BDTG",basepath%"el_pteta_high_ec",_CommonSpect,elVars) ),
        ])
    def __call__(self,lep,ncorr="auto"):
        if ncorr == "auto": ncorr = 0 # (1 if self._isMC else 0)
        if   abs(lep.pdgId()) == 11: return self.el(lep,ncorr)
        elif abs(lep.pdgId()) == 13: return self.mu(lep,ncorr)
        else: return -99

