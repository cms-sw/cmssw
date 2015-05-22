#!/usr/bin/env python
from CMGTools.TTHAnalysis.treeReAnalyzer import *
from CMGTools.TTHAnalysis.tools.mvaTool import *
from math import *
def safelog(x):
    return log(x) if x > 0 else -99

_Spect = [
    MVAVar("pt := LepGood_pt", func = lambda LepGood : LepGood.pt),
    MVAVar("tightId := LepGood_tightId", func = lambda LepGood : LepGood.tightId),
]
_Vars = {
    'BPH':[ 
        MVAVar("eta := LepGood_eta", func = lambda LepGood : LepGood.eta),
        MVAVar("globalTrackChi2 := log(LepGood_globalTrackChi2)", func = lambda LepGood : safelog(LepGood.globalTrackChi2)),
        MVAVar("segmentCompatibility := LepGood_segmentCompatibility", func = lambda LepGood : LepGood.segmentCompatibility),
        MVAVar("chi2LocalPosition := log(LepGood_chi2LocalPosition)", func = lambda LepGood : safelog(LepGood.chi2LocalPosition)),
        MVAVar("chi2LocalMomentum := log(LepGood_chi2LocalMomentum)", func = lambda LepGood : safelog(LepGood.chi2LocalMomentum)),
        MVAVar("innerTrackValidHitFraction := LepGood_innerTrackValidHitFraction", func = lambda LepGood : LepGood.innerTrackValidHitFraction),
        MVAVar("lostOuterHits := LepGood_lostOuterHits", func = lambda LepGood : LepGood.lostOuterHits),
        MVAVar("glbTrackProbability := log(LepGood_glbTrackProbability)", func = lambda LepGood : safelog(LepGood.glbTrackProbability)),
        MVAVar("trackerHits := LepGood_trackerHits", func = lambda LepGood : LepGood.trackerHits),
    ],
    'Calo':[
        MVAVar("caloCompatibility := LepGood_caloCompatibility", func = lambda LepGood : LepGood.caloCompatibility),
        MVAVar("caloEMEnergy := min(LepGood_caloEMEnergy,20)", func = lambda LepGood : min(LepGood.caloEMEnergy,20)),
        MVAVar("caloHadEnergy := min(LepGood_caloHadEnergy,30)", func = lambda LepGood : min(LepGood.caloHadEnergy,30)),
    ],
    'Trk':[
        MVAVar("lostHits := LepGood_lostHits", func = lambda LepGood : LepGood.lostHits),
        MVAVar("trkKink := min(100,LepGood_trkKink)", func = lambda LepGood : min(100,LepGood.trkKink)),
        MVAVar("trackerLayers := LepGood_trackerLayers", func = lambda LepGood : LepGood.trackerLayers),
        MVAVar("pixelLayers := LepGood_pixelLayers", func = lambda LepGood : LepGood.pixelLayers),
        MVAVar("innerTrackChi2 := LepGood_innerTrackChi2", func = lambda LepGood : LepGood.innerTrackChi2),
    ]
}

class MuonMVA:
    def __init__(self,name,path):
        global _Vars, _Spect
        if name == "BPH":
            vars = _Vars["BPH"]
        if name == "BPHCalo":
            vars = _Vars["BPH"] + _Vars['Calo']
        if name == "Full":
            vars = _Vars["BPH"] + _Vars['Calo'] + _Vars['Trk']
        self.mva = MVATool("BDTG",path,vars,specs=_Spect,rarity=True)
    def __call__(self,lep):
        if   abs(lep.pdgId) == 11: return 1.0
        elif abs(lep.pdgId) == 13: return self.mva(lep)
        else: return -99

class MuonMVAFriend:
    def __init__(self,name,path,label=""):
        self.mva = MuonMVA(name,path) 
        self.label = label
    def listBranches(self):
        return [ ("nLepGood","I"), ("LepGood_muonMVAId"+self.label,"F",8,"nLepGood") ]
    def __call__(self,event):
        lep = Collection(event,"LepGood","nLepGood",8)
        ret = { 'nLepGood' : event.nLepGood }
        ret['LepGood_muonMVAId'+self.label] = [ self.mva(l) for l in lep ] 
        return ret

if __name__ == '__main__':
    from sys import argv
    file = ROOT.TFile(argv[1])
    tree = file.Get("treeProducerSusyMultilepton")
    tree.vectorTree = True
    class Tester(Module):
        def __init__(self, name):
            Module.__init__(self,name,None)
            self.bph  = MuonMVAFriend("BPH",     "/afs/cern.ch/work/g/gpetrucc/micro/cmg/CMSSW_7_0_9/src/CMGTools/TTHAnalysis/macros/leptons/train70XBPH_BDTG.weights.xml")
            self.bphc = MuonMVAFriend("BPHCalo", "/afs/cern.ch/work/g/gpetrucc/micro/cmg/CMSSW_7_0_9/src/CMGTools/TTHAnalysis/macros/leptons/train70XBPHCalo_BDTG.weights.xml")
            self.full = MuonMVAFriend("Full",    "/afs/cern.ch/work/g/gpetrucc/micro/cmg/CMSSW_7_0_9/src/CMGTools/TTHAnalysis/macros/leptons/train70XFull_BDTG.weights.xml")
        def analyze(self,ev):
            print "\nrun %6d lumi %4d event %d: leps %d" % (ev.run, ev.lumi, ev.evt, ev.nLepGood)
            ret = self.full(ev)
            print ret
            lep = Collection(ev,"LepGood","nLepGood",8)
            for i,l in enumerate(lep):
                if abs(l.pdgId) == 11: continue
                print "Lepton %d (pdgId %+2d): mvaId =  +%.4f   mvaIdFriend = +%.4f" % (i, l.pdgId, l.mvaId, ret['LepGood_muonMVAId'][i])     
            
    el = EventLoop([ Tester("tester") ])
    el.loop([tree], maxEvents = 50)

        
