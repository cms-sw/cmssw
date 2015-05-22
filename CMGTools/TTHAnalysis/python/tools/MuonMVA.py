#!/usr/bin/env python
from CMGTools.TTHAnalysis.treeReAnalyzer import *
from CMGTools.TTHAnalysis.tools.mvaTool import *
from math import *
def safelog(x):
    return log(x) if x > 0 else -99

_Spect = [
    MVAVar("pt := LepGood_pt", func = lambda LepGood : LepGood.pt()),
    MVAVar("tightId := LepGood_tightId", func = lambda LepGood : LepGood.tightId()),
]
_Vars = {
    'BPH':[ 
        MVAVar("eta := LepGood_eta", func = lambda LepGood : LepGood.eta()),
        MVAVar("globalTrackChi2 := log(LepGood_globalTrackChi2)", func = lambda LepGood : safelog(LepGood.globalTrack().normalizedChi2() if LepGood.globalTrack().isNonnull() else 0)),
        MVAVar("segmentCompatibility := LepGood_segmentCompatibility", func = lambda LepGood : LepGood.segmentCompatibility()),
        MVAVar("chi2LocalPosition := log(LepGood_chi2LocalPosition)", func = lambda LepGood : safelog(LepGood.combinedQuality().chi2LocalPosition)),
        MVAVar("chi2LocalMomentum := log(LepGood_chi2LocalMomentum)", func = lambda LepGood : safelog(LepGood.combinedQuality().chi2LocalMomentum)),
        MVAVar("innerTrackValidHitFraction := LepGood_innerTrackValidHitFraction", func = lambda LepGood : LepGood.innerTrack().validFraction()),
        MVAVar("lostOuterHits := LepGood_lostOuterHits", func = lambda LepGood : LepGood.innerTrack().hitPattern().numberOfLostHits(ROOT.reco.HitPattern.MISSING_OUTER_HITS)),
        MVAVar("glbTrackProbability := log(LepGood_glbTrackProbability)", func = lambda LepGood : safelog(LepGood.combinedQuality().glbTrackProbability)),
        MVAVar("trackerHits := LepGood_trackerHits", func = lambda LepGood : LepGood.track().hitPattern().numberOfValidTrackerHits()),
    ],
    'Calo':[
        MVAVar("caloCompatibility := LepGood_caloCompatibility", func = lambda LepGood : LepGood.caloCompatibility()),
        MVAVar("caloEMEnergy := min(LepGood_caloEMEnergy,20)", func = lambda LepGood : min(LepGood.calEnergy().em,20)),
        MVAVar("caloHadEnergy := min(LepGood_caloHadEnergy,30)", func = lambda LepGood : min(LepGood.calEnergy().had,30)),
    ],
    'Trk':[
        MVAVar("lostHits := LepGood_lostHits", func = lambda LepGood : LepGood.innerTrack().hitPattern().numberOfLostHits(ROOT.reco.HitPattern.MISSING_INNER_HITS)),
        MVAVar("trkKink := min(100,LepGood_trkKink)", func = lambda LepGood : min(100,LepGood.combinedQuality().trkKink)),
        MVAVar("trackerLayers := LepGood_trackerLayers", func = lambda LepGood : LepGood.track().hitPattern().trackerLayersWithMeasurement()),
        MVAVar("pixelLayers := LepGood_pixelLayers", func = lambda LepGood : LepGood.track().hitPattern().pixelLayersWithMeasurement()),
        MVAVar("innerTrackChi2 := LepGood_innerTrackChi2", func = lambda LepGood : LepGood.innerTrack().normalizedChi2()),
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
        if   abs(lep.pdgId()) == 11: return 1.0
        elif abs(lep.pdgId()) == 13: 
            if lep.innerTrack().isNonnull() and lep.isPFMuon():
                return self.mva(lep)
            else:
                return -1;
        else: return -99
        
