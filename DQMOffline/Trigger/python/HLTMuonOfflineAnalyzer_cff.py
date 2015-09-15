import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.HLTMuonOfflineAnalyzer_cfi import hltMuonOfflineAnalyzer

globalMuonParams = cms.PSet(
    d0Cut = cms.untracked.double(2.0),
    z0Cut = cms.untracked.double(25.0),
    recoCuts = cms.untracked.string("obj.isGlobalMuon() && std::abs(obj.eta()) < 2.4"),
    hltCuts  = cms.untracked.string("std::abs(obj.eta()) < 2.4"),
)

trackerMuonParams = cms.PSet(
    d0Cut = cms.untracked.double(2.0),
    z0Cut = cms.untracked.double(25.0),
    recoCuts = cms.untracked.string("obj.isTrackerMuon() && std::abs(obj.eta()) < 2.4"),
    hltCuts  = cms.untracked.string("std::abs(obj.eta()) < 2.4"),
)


tightMuonParams = cms.PSet(
    d0Cut = cms.untracked.double(0.2),
    z0Cut = cms.untracked.double(0.5),
    recoCuts = cms.untracked.string(' && '.join([
        "obj.isGlobalMuon() && obj.isPFMuon() && obj.isTrackerMuon()",
        "std::abs(obj.eta()) < 2.4",
        "obj.innerTrack()->hitPattern().numberOfValidPixelHits() > 0",
        "obj.innerTrack()->hitPattern().trackerLayersWithMeasurement() > 5",
        "(obj.pfIsolationR04().sumChargedHadronPt + std::max(obj.pfIsolationR04().sumNeutralHadronEt + obj.pfIsolationR04().sumPhotonEt - obj.pfIsolationR04().sumPUPt/2.0,0.0))/obj.pt() < 0.12", 
        "obj.globalTrack()->hitPattern().numberOfValidMuonHits() > 0",
        "obj.globalTrack()->normalizedChi2() < 10",
        "obj.numberOfMatches() > 1"
        ])),
    hltCuts  = cms.untracked.string("std::abs(obj.eta()) < 2.4"),
)



looseMuonParams = cms.PSet(
    d0Cut = cms.untracked.double(50),
    z0Cut = cms.untracked.double(100),
    recoCuts = cms.untracked.string(' && '.join([
        "obj.isPFMuon() && (obj.isTrackerMuon() || obj.isGlobalMuon())",
        "(obj.pfIsolationR04().sumChargedHadronPt + std::max(obj.pfIsolationR04().sumNeutralHadronEt + obj.pfIsolationR04().sumPhotonEt - obj.pfIsolationR04().sumPUPt/2.0,0.0))/obj.pt() < 0.20"
        ])),
    hltCuts  = cms.untracked.string("std::abs(obj.eta()) < 2.4"),
)


globalAnalyzer = hltMuonOfflineAnalyzer.clone()
globalAnalyzer.destination = "HLT/Muon/DistributionsGlobal"
globalAnalyzer.targetParams = globalMuonParams
#globalAnalyzer.probeParams = cms.PSet()

trackerAnalyzer = hltMuonOfflineAnalyzer.clone()
trackerAnalyzer.destination = "HLT/Muon/DistributionsTracker"
trackerAnalyzer.targetParams = trackerMuonParams
#trackerAnalyzer.probeParams = cms.PSet()

tightAnalyzer = hltMuonOfflineAnalyzer.clone()
tightAnalyzer.destination = "HLT/Muon/DistributionsTight"
tightAnalyzer.targetParams = tightMuonParams
#tightAnalyzer.probeParams = cms.PSet() 

looseAnalyzer = hltMuonOfflineAnalyzer.clone()
looseAnalyzer.destination = "HLT/Muon/DistributionsLoose"
looseAnalyzer.targetParams = looseMuonParams
#tightAnalyzer.probeParams = cms.PSet() 



hltMuonOfflineAnalyzers = cms.Sequence(
    globalAnalyzer *
    trackerAnalyzer *
    tightAnalyzer *
    looseAnalyzer
)
