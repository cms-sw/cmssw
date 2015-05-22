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
    MVAVar("LepGood_neuRelIso03 := LepGood_relIso03 - LepGood_chargedHadRelIso03",lambda x: x.relIso03 - x.chargedHadRelIso03),  
    MVAVar("LepGood_chRelIso03 := LepGood_chargedHadRelIso03",lambda x: x.chargedHadRelIso03),
    MVAVar("LepGood_jetDR := min(LepGood_jetDR,0.5)", lambda x : min(x.jetDR,0.5), corrfunc=ROOT.correctJetDRMC),
    MVAVar("LepGood_jetPtRatio := min(LepGood_jetPtRatio,1.5)", lambda x : min(x.jetPtRatio,1.5), corrfunc=ROOT.correctJetPtRatioMC),
    MVAVar("LepGood_jetBTagCSV := max(LepGood_jetBTagCSV,0)", lambda x : max(x.jetBTagCSV,0.)),
    MVAVar("LepGood_sip3d",lambda x: x.sip3d, corrfunc=ROOT.scaleSip3dMC),
    MVAVar("LepGood_dxy := log(abs(LepGood_dxy))",lambda x: log(abs(x.dxy)), corrfunc=ROOT.scaleDxyMC),
    MVAVar("LepGood_dz  := log(abs(LepGood_dz))", lambda x: log(abs(x.dz)), corrfunc=ROOT.scaleDzMC),
]

_ElectronVars = [
    MVAVar("LepGood_mvaId",lambda x: x.mvaId)
]

_MuonVars = [
    MVAVar("LepGood_segmentCompatibility",lambda x: x.segmentCompatibility)
]

_SVVars = [
    MVAVar("LepGood_hasSV", lambda x: x.hasSV),
    MVAVar("LepGood_svRedPt := min(max( LepGood_svRedPt, -10),100)", lambda x : min(max( x.svRedPt, -10),100)),
    MVAVar("LepGood_svRedM := min(max( LepGood_svRedM, -1),6)", lambda x : min(max( x.svRedM, -1),6)),
    MVAVar("LepGood_svM := min(max( LepGood_svM, -1),6)", lambda x : min(max( x.svM, -1),6)),
    MVAVar("LepGood_svPt := min(max( LepGood_svPt, -10),100)", lambda x : min(max( x.svPt, -10),100)),
    MVAVar("LepGood_svSip3d := min(max( LepGood_svSip3d, -10),100)", lambda x : min(max( x.svSip3d, -10),100)),
    MVAVar("LepGood_svLepSip3d := min(max( LepGood_svLepSip3d, -10),5)", lambda x : min(max( x.svLepSip3d, -10),5)),
    MVAVar("LepGood_svNTracks := min(max( LepGood_svNTracks, -1),10)", lambda x : min(max( x.svNTracks, -1),10)),
    MVAVar("LepGood_svChi2n := min(max( LepGood_svChi2n, -1),10)", lambda x : min(max( x.svChi2n, -1),10)),
    MVAVar("LepGood_svDxy := min(max( LepGood_svDxy, 0),4)", lambda x : min(max( x.svDxy, 0),4)),
]
class LeptonMVA:
    def __init__(self,basepath,training="susyBase"):
        global _CommonVars, _CommonSpect, _ElectronVars, _MuonVars, _SVVars
        if type(basepath) == tuple: basepathmu, basepathel  = basepath
        else:                       basepathmu, basepathel  = basepath, basepath
        if training == "muMVAId":
            muVars = _CommonVars[:] + [ MVAVar("mvaId",lambda x: x.muonMVAIdFull) ]
            elVars = _CommonVars[:] + _ElectronVars[:]
        elif training == "muMVAId_SV":
            muVars = _CommonVars[:] + [ MVAVar("mvaId",lambda x: x.muonMVAIdFull) ] + _SVVars
            elVars = _CommonVars[:] + _ElectronVars[:] + _SVVars
        else:
            muVars = _CommonVars[:] + _MuonVars[:]
            elVars = _CommonVars[:] + _ElectronVars[:]
        self.mu = CategorizedMVA([
            ( lambda x: x.pt <= 10, MVATool("BDTG",basepathmu%"mu_pteta_low", _CommonSpect,muVars) ),
            ( lambda x: x.pt > 10 and x.pt <= 25 and abs(x.eta) <  1.5 , MVATool("BDTG",basepathmu%"mu_pteta_medium_b", _CommonSpect,muVars) ),
            ( lambda x: x.pt > 10 and x.pt <= 25 and abs(x.eta) >= 1.5 , MVATool("BDTG",basepathmu%"mu_pteta_medium_e", _CommonSpect,muVars) ),
            ( lambda x: x.pt >  25 and abs(x.eta) <  1.5 , MVATool("BDTG",basepathmu%"mu_pteta_high_b",_CommonSpect,muVars) ),
            ( lambda x: x.pt >  25 and abs(x.eta) >= 1.5 , MVATool("BDTG",basepathmu%"mu_pteta_high_e",_CommonSpect,muVars) ),
        ])
        self.el = CategorizedMVA([
            ( lambda x: x.pt <= 10                          , MVATool("BDTG",basepathel%"el_pteta_low", _CommonSpect,elVars) ),    
            ( lambda x: x.pt >  10 and x.pt <= 25 and abs(x.eta) <  0.8                         , MVATool("BDTG",basepathel%"el_pteta_medium_cb", _CommonSpect,elVars) ),
            ( lambda x: x.pt >  10 and x.pt <= 25 and abs(x.eta) >= 0.8 and abs(x.eta) <  1.479 , MVATool("BDTG",basepathel%"el_pteta_medium_fb", _CommonSpect,elVars) ),
            ( lambda x: x.pt >  10 and x.pt <= 25 and abs(x.eta) >= 1.479                       , MVATool("BDTG",basepathel%"el_pteta_medium_ec", _CommonSpect,elVars) ),
            ( lambda x: x.pt >  25 and abs(x.eta) <  0.8                         , MVATool("BDTG",basepathel%"el_pteta_high_cb",_CommonSpect,elVars) ),
            ( lambda x: x.pt >  25 and abs(x.eta) >= 0.8 and abs(x.eta) <  1.479 , MVATool("BDTG",basepathel%"el_pteta_high_fb",_CommonSpect,elVars) ),
            ( lambda x: x.pt >  25 and abs(x.eta) >= 1.479                       , MVATool("BDTG",basepathel%"el_pteta_high_ec",_CommonSpect,elVars) ),
        ])
    def __call__(self,lep,ncorr=0):
        if   abs(lep.pdgId) == 11: return self.el(lep,ncorr)
        elif abs(lep.pdgId) == 13: return self.mu(lep,ncorr)
        else: return -99

class LepMVAFriend:
    def __init__(self,path,training="susyBase",label="",fast=True):
        self.mva = LeptonMVA(path+"/weights/%s_BDTG.weights.xml" if type(path) == str else path, training=training)
        self.fast = fast
        self.label = label
    def listBranches(self):
        return [ ("nLepGood","I"), ("LepGood_mvaSusyPHYS14"+self.label,"F",8,"nLepGood") ]
    def __call__(self,event):
        lep = Collection(event,"LepGood","nLepGood",8)
        ret = { 'nLepGood' : event.nLepGood }
        if event.run >= 1: # DATA
            ret['LepGood_mvaSusyPHYS14'+self.label] = [ self.mva(l, ncorr=0) for l in lep ] 
        else:              # MC
            ret['LepGood_mvaSusyPHYS14'+self.label] = [ self.mva(l, ncorr=0) for l in lep ] 
        return ret

if __name__ == '__main__':
    from sys import argv
    file = ROOT.TFile(argv[1])
    tree = file.Get("tree")
    if len(argv) > 2: tree.AddFriend("sf/t",argv[2])
    tree.vectorTree = True
    class Tester(Module):
        def __init__(self, name):
            Module.__init__(self,name,None)
            ## Vanilla training (weights originally from /afs/cern.ch/user/b/botta/CMGToolsGit/newRecipe70/CMSSW_7_0_6_patch1/src/CMGTools/TTHAnalysis/macros/leptons/weights_SUSY_presel 
            self.sf0  = LepMVAFriend(("/afs/cern.ch/work/g/gpetrucc/TREES_70X_240914/0_lepMVA_v1/%s_BDTG.weights.xml",
                                      "/afs/cern.ch/work/g/gpetrucc/TREES_70X_240914/0_lepMVA_v1/%s_BDTG.weights.xml",))
            ## Trial training with MVA Muon ID and SV variables
            self.sf2  = LepMVAFriend(("/afs/cern.ch/work/g/gpetrucc/TREES_70X_240914/0_lepMVA_v1/SV_%s_BDTG.weights.xml",
                                      "/afs/cern.ch/work/g/gpetrucc/TREES_70X_240914/0_lepMVA_v1/SV_%s_BDTG.weights.xml",),
                                     training="muMVAId_SV", label="SV")
        def analyze(self,ev):
            print "\nrun %6d lumi %4d event %d: leps %d" % (ev.run, ev.lumi, ev.evt, ev.nLepGood)
            print self.sf0(ev)
            print self.sf2(ev)
    el = EventLoop([ Tester("tester") ])
    el.loop([tree], maxEvents = 50)

        
