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


refPathsList = cms.vstring(
    "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v",
    "HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v",
    "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v",
    "HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_v",
    "HLT_TkMu17_TrkIsoVVL_TkMu8_TrkIsoVVL_v",
    "HLT_TkMu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v"
    "HLT_Mu18_Mu9_DZ_v",
    "HLT_Mu18_Mu9_v",
    "HLT_Mu18_Mu9_SameSign_DZ_v",
    "HLT_Mu18_Mu9_SameSign_v"
    )

globalAnalyzerRef = hltMuonOfflineAnalyzer.clone()
globalAnalyzerRef.destination = "HLT/Muon/DistributionsGlobal"
globalAnalyzerRef.targetParams = globalMuonParams
globalAnalyzerRef.hltPathsToCheck = refPathsList
globalAnalyzerRef.requiredTriggers = cms.untracked.vstring(
    "HLT_Mu17_TrkIsoVVL_v")

#globalAnalyzerRef.probeParams = cms.PSet()

trackerAnalyzerRef = hltMuonOfflineAnalyzer.clone()
trackerAnalyzerRef.destination = "HLT/Muon/DistributionsTracker"
trackerAnalyzerRef.targetParams = trackerMuonParams
trackerAnalyzerRef.hltPathsToCheck = refPathsList
trackerAnalyzerRef.requiredTriggers = cms.untracked.vstring(
    "HLT_Mu17_TrkIsoVVL_v")
#trackerAnalyzerRef.probeParams = cms.PSet()

tightAnalyzerRef = hltMuonOfflineAnalyzer.clone()
tightAnalyzerRef.destination = "HLT/Muon/DistributionsTight"
tightAnalyzerRef.targetParams = tightMuonParams
tightAnalyzerRef.hltPathsToCheck = refPathsList

tightAnalyzerRef.requiredTriggers = cms.untracked.vstring(
    "HLT_Mu17_TrkIsoVVL_v")
#tightAnalyzerRef.probeParams = cms.PSet() 

looseAnalyzerRef = hltMuonOfflineAnalyzer.clone()
looseAnalyzerRef.destination = "HLT/Muon/DistributionsLoose"
looseAnalyzerRef.targetParams = looseMuonParams
looseAnalyzerRef.hltPathsToCheck = refPathsList
looseAnalyzerRef.requiredTriggers = cms.untracked.vstring(
    "HLT_Mu17_TrkIsoVVL_v")
#tightAnalyzer.probeParams = cms.PSet() 



refPathsList19 =  cms.vstring(
    "HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_Mass3p8_v", 
    "HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_Mass8_v"  ,
    "HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ_v"        ,
    "HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_v",
    "HLT_Mu20_Mu10_DZ_v",
    "HLT_Mu20_Mu10_SameSign_DZ_v",
    "HLT_Mu20_Mu10_SameSign_v",
    "HLT_Mu20_Mu10_v",
    "HLT_Mu23_Mu12_DZ_v",
    "HLT_Mu23_Mu12_SameSign_DZ_v",
    "HLT_Mu23_Mu12_SameSign_v",
    "HLT_Mu23_Mu12_v"
    )

globalAnalyzerRef19 = hltMuonOfflineAnalyzer.clone()
globalAnalyzerRef19.destination = "HLT/Muon/DistributionsGlobal"
globalAnalyzerRef19.targetParams = globalMuonParams
globalAnalyzerRef19.hltPathsToCheck = refPathsList19
globalAnalyzerRef19.requiredTriggers = cms.untracked.vstring(
    "HLT_Mu19_TrkIsoVVL_v")

#globalAnalyzerRef19.probeParams = cms.PSet()

trackerAnalyzerRef19 = hltMuonOfflineAnalyzer.clone()
trackerAnalyzerRef19.destination = "HLT/Muon/DistributionsTracker"
trackerAnalyzerRef19.targetParams = trackerMuonParams
trackerAnalyzerRef19.hltPathsToCheck = refPathsList19
trackerAnalyzerRef19.requiredTriggers = cms.untracked.vstring(
    "HLT_Mu19_TrkIsoVVL_v")
#trackerAnalyzerRef19.probeParams = cms.PSet()

tightAnalyzerRef19 = hltMuonOfflineAnalyzer.clone()
tightAnalyzerRef19.destination = "HLT/Muon/DistributionsTight"
tightAnalyzerRef19.targetParams = tightMuonParams
tightAnalyzerRef19.hltPathsToCheck = refPathsList19
tightAnalyzerRef19.requiredTriggers = cms.untracked.vstring(
    "HLT_Mu19_TrkIsoVVL_v")
#tightAnalyzerRef19.probeParams = cms.PSet() 

looseAnalyzerRef19 = hltMuonOfflineAnalyzer.clone()
looseAnalyzerRef19.destination = "HLT/Muon/DistributionsLoose"
looseAnalyzerRef19.targetParams = looseMuonParams
looseAnalyzerRef19.hltPathsToCheck = refPathsList19
looseAnalyzerRef19.requiredTriggers = cms.untracked.vstring(
    "HLT_Mu19_TrkIsoVVL_v")
#tightAnalyzer.probeParams = cms.PSet() 




hltMuonOfflineAnalyzers = cms.Sequence(
    globalAnalyzerTnP  *
    trackerAnalyzerTnP *
    tightAnalyzerTnP   *
    looseAnalyzerTnP   *
    globalAnalyzerRef  *
    trackerAnalyzerRef *
    tightAnalyzerRef   *
    looseAnalyzerRef   *
    globalAnalyzerRef19  *
    trackerAnalyzerRef19 *
    tightAnalyzerRef19   *
    looseAnalyzerRef19
)

from Configuration.Eras.Modifier_pA_2016_cff import pA_2016
from Configuration.Eras.Modifier_ppRef_2017_cff import ppRef_2017
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
for muAna in [globalAnalyzerTnP.targetParams, trackerAnalyzerTnP.targetParams, 
              tightAnalyzerTnP.targetParams, looseAnalyzerTnP.targetParams,
              globalAnalyzerRef.targetParams, trackerAnalyzerRef.targetParams, 
              tightAnalyzerRef.targetParams, looseAnalyzerRef.targetParams,]:
    for e in [pA_2016, ppRef_2017, pp_on_AA]:
	    e.toModify(muAna, ptCut_Jpsi = cms.untracked.double( 5.0))
for muAna in [globalAnalyzerTnP.binParams, trackerAnalyzerTnP.binParams,
              tightAnalyzerTnP.binParams, looseAnalyzerTnP.binParams,
              globalAnalyzerRef.binParams, trackerAnalyzerRef.binParams,
              tightAnalyzerRef.binParams, looseAnalyzerRef.binParams]:
    for e in [pA_2016, ppRef_2017, pp_on_AA]:
	    e.toModify(muAna, ptCoarse = cms.untracked.vdouble(0.,1.,2.,3.,4.,5.,7.,9.,12.,15.,20.,30.,40.))
