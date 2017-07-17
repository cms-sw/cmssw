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
        "(pfIsolationR04().sumChargedHadronPt + max(pfIsolationR04().sumNeutralHadronEt + pfIsolationR04().sumPhotonEt - pfIsolationR04().sumPUPt/2,0.0))/pt < 0.15", 
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
        "(pfIsolationR04().sumChargedHadronPt + max(pfIsolationR04().sumNeutralHadronEt + pfIsolationR04().sumPhotonEt - pfIsolationR04().sumPUPt/2,0.0))/pt < 0.25"
        ])),
    hltCuts  = cms.untracked.string("abs(eta) < 2.4"),
)


globalAnalyzerTnP = hltMuonOfflineAnalyzer.clone()
globalAnalyzerTnP.destination = "HLT/Muon/DistributionsGlobal"
globalAnalyzerTnP.targetParams = globalMuonParams
#globalAnalyzerTnP.probeParams = cms.PSet()

trackerAnalyzerTnP = hltMuonOfflineAnalyzer.clone()
trackerAnalyzerTnP.destination = "HLT/Muon/DistributionsTracker"
trackerAnalyzerTnP.targetParams = trackerMuonParams
#trackerAnalyzerTnP.probeParams = cms.PSet()

tightAnalyzerTnP = hltMuonOfflineAnalyzer.clone()
tightAnalyzerTnP.destination = "HLT/Muon/DistributionsTight"
tightAnalyzerTnP.targetParams = tightMuonParams
#tightAnalyzerTnP.probeParams = cms.PSet() 

looseAnalyzerTnP = hltMuonOfflineAnalyzer.clone()
looseAnalyzerTnP.destination = "HLT/Muon/DistributionsLoose"
looseAnalyzerTnP.targetParams = looseMuonParams
#tightAnalyzer.probeParams = cms.PSet() 


globalAnalyzerRef = hltMuonOfflineAnalyzer.clone()
globalAnalyzerRef.destination = "HLT/Muon/DistributionsGlobal"
globalAnalyzerRef.targetParams = globalMuonParams
globalAnalyzerRef.hltPathsToCheck = cms.vstring(
    "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v",
    "HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v",
    "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v",
    "HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_v",
    "HLT_TkMu17_TrkIsoVVL_TkMu8_TrkIsoVVL_v",
    "HLT_TkMu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v")
globalAnalyzerRef.requiredTriggers = cms.untracked.vstring(
    "HLT_Mu17_TrkIsoVVL_v")

#globalAnalyzerRef.probeParams = cms.PSet()

trackerAnalyzerRef = hltMuonOfflineAnalyzer.clone()
trackerAnalyzerRef.destination = "HLT/Muon/DistributionsTracker"
trackerAnalyzerRef.targetParams = trackerMuonParams
trackerAnalyzerRef.hltPathsToCheck = cms.vstring(
    "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v",
    "HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v",
    "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v",
    "HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_v",
    "HLT_TkMu17_TrkIsoVVL_TkMu8_TrkIsoVVL_v",
    "HLT_TkMu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v")
trackerAnalyzerRef.requiredTriggers = cms.untracked.vstring(
    "HLT_Mu17_TrkIsoVVL_v")
#trackerAnalyzerRef.probeParams = cms.PSet()

tightAnalyzerRef = hltMuonOfflineAnalyzer.clone()
tightAnalyzerRef.destination = "HLT/Muon/DistributionsTight"
tightAnalyzerRef.targetParams = tightMuonParams
tightAnalyzerRef.hltPathsToCheck = cms.vstring(
    "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v",
    "HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v",
    "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v",
    "HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_v",
    "HLT_TkMu17_TrkIsoVVL_TkMu8_TrkIsoVVL_v",
    "HLT_TkMu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v")
tightAnalyzerRef.requiredTriggers = cms.untracked.vstring(
    "HLT_Mu17_TrkIsoVVL_v")
#tightAnalyzerRef.probeParams = cms.PSet() 

looseAnalyzerRef = hltMuonOfflineAnalyzer.clone()
looseAnalyzerRef.destination = "HLT/Muon/DistributionsLoose"
looseAnalyzerRef.targetParams = looseMuonParams
looseAnalyzerRef.hltPathsToCheck = cms.vstring(
    "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v",
    "HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v",
    "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v",
    "HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_v",
    "HLT_TkMu17_TrkIsoVVL_TkMu8_TrkIsoVVL_v",
    "HLT_TkMu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v")
looseAnalyzerRef.requiredTriggers = cms.untracked.vstring(
    "HLT_Mu17_TrkIsoVVL_v")
#tightAnalyzer.probeParams = cms.PSet() 




hltMuonOfflineAnalyzers = cms.Sequence(
    globalAnalyzerTnP  *
    trackerAnalyzerTnP *
    tightAnalyzerTnP   *
    looseAnalyzerTnP   *
    globalAnalyzerRef  *
    trackerAnalyzerRef *
    tightAnalyzerRef   *
    looseAnalyzerRef
)

from Configuration.Eras.Modifier_pA_2016_cff import pA_2016
for muAna in [globalAnalyzerTnP.targetParams, trackerAnalyzerTnP.targetParams, 
              tightAnalyzerTnP.targetParams, looseAnalyzerTnP.targetParams,
              globalAnalyzerRef.targetParams, trackerAnalyzerRef.targetParams, 
              tightAnalyzerRef.targetParams, looseAnalyzerRef.targetParams,]:
    pA_2016.toModify(muAna, ptCut_Jpsi = cms.untracked.double( 5.0))
for muAna in [globalAnalyzerTnP.binParams, trackerAnalyzerTnP.binParams,
              tightAnalyzerTnP.binParams, looseAnalyzerTnP.binParams,
              globalAnalyzerRef.binParams, trackerAnalyzerRef.binParams,
              tightAnalyzerRef.binParams, looseAnalyzerRef.binParams]:
    pA_2016.toModify(muAna, ptCoarse = cms.untracked.vdouble(0.,1.,2.,3.,4.,5.,7.,9.,12.,15.,20.,30.,40.))
