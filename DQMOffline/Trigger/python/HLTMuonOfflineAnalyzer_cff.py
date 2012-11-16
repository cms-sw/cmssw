import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.HLTMuonOfflineAnalyzer_cfi import hltMuonOfflineAnalyzer

globalMuonParams = cms.PSet(
    d0Cut = cms.untracked.double(2.0),
    z0Cut = cms.untracked.double(25.0),
    recoCuts = cms.untracked.string("isGlobalMuon && abs(eta) < 2.4"),
    hltCuts  = cms.untracked.string("abs(eta) < 2.4"),
)

trackerMuonParams = cms.PSet(
    d0Cut = cms.untracked.double(2.0),
    z0Cut = cms.untracked.double(25.0),
    recoCuts = cms.untracked.string("isTrackerMuon && abs(eta) < 2.4"),
    hltCuts  = cms.untracked.string("abs(eta) < 2.4"),
)


tightMuonParams = cms.PSet(
    d0Cut = cms.untracked.double(0.2),
    z0Cut = cms.untracked.double(0.5),
    recoCuts = cms.untracked.string(' && '.join([
        "isGlobalMuon && isPFMuon && isTrackerMuon",
        "abs(eta) < 2.4",
        "innerTrack.hitPattern.numberOfValidPixelHits > 0",
        "innerTrack.hitPattern.trackerLayersWithMeasurement > 5",
        "(pfIsolationR04().sumChargedHadronPt + max(pfIsolationR04().sumNeutralHadronEt + pfIsolationR04().sumPhotonEt - pfIsolationR04().sumPUPt/2,0.0))/pt < 0.12", 
        "globalTrack.hitPattern.numberOfValidMuonHits > 0",
        "globalTrack.normalizedChi2 < 10",
        "numberOfMatches > 1"
        ])),
    hltCuts  = cms.untracked.string("abs(eta) < 2.4"),
)



looseMuonParams = cms.PSet(
    d0Cut = cms.untracked.double(50),
    z0Cut = cms.untracked.double(100),
    recoCuts = cms.untracked.string(' && '.join([
        "isPFMuon && (isTrackerMuon || isGlobalMuon)",
        "(pfIsolationR04().sumChargedHadronPt + max(pfIsolationR04().sumNeutralHadronEt + pfIsolationR04().sumPhotonEt - pfIsolationR04().sumPUPt/2,0.0))/pt < 0.20"
        ])),
    hltCuts  = cms.untracked.string("abs(eta) < 2.4"),
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
