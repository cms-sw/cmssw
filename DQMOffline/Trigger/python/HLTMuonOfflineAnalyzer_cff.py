import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.HLTMuonOfflineAnalyzer_cfi import hltMuonOfflineAnalyzer

globalMuonParams = cms.PSet(
    d0Cut = cms.untracked.double(2.0),
    z0Cut = cms.untracked.double(25.0),
    recoMaxEtaCut = cms.untracked.double(2.4),
    recoMinEtaCut = cms.untracked.double(0.0),
    recoGlbMuCut = cms.untracked.bool(True),
    hltMaxEtaCut  = cms.untracked.double(2.4),
    hltMinEtaCut  = cms.untracked.double(0.0),
)

globalAnalyzerTnP = hltMuonOfflineAnalyzer.clone()
globalAnalyzerTnP.destination = "HLT/Muon/DistributionsGlobal"
globalAnalyzerTnP.targetParams = globalMuonParams
#globalAnalyzerTnP.probeParams = cms.PSet()

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

refPathsList19 =  cms.vstring(
    "HLT_Mu23_Mu12_DZ_v",
    "HLT_Mu23_Mu12_v"
    )

globalAnalyzerRef19 = hltMuonOfflineAnalyzer.clone()
globalAnalyzerRef19.destination = "HLT/Muon/DistributionsGlobal"
globalAnalyzerRef19.targetParams = globalMuonParams
globalAnalyzerRef19.hltPathsToCheck = refPathsList19
globalAnalyzerRef19.requiredTriggers = cms.untracked.vstring(
    "HLT_Mu19_TrkIsoVVL_v")

#globalAnalyzerRef19.probeParams = cms.PSet()

hltMuonOfflineAnalyzers = cms.Sequence(
    globalAnalyzerTnP  *
    globalAnalyzerRef  *
    globalAnalyzerRef19
)

from Configuration.Eras.Modifier_pA_2016_cff import pA_2016
from Configuration.Eras.Modifier_ppRef_2017_cff import ppRef_2017
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA

for muAna in [globalAnalyzerTnP.targetParams, 
              globalAnalyzerRef.targetParams]:
    for e in [pA_2016, ppRef_2017, pp_on_AA]:
	    e.toModify(muAna, ptCut_Jpsi = cms.untracked.double( 5.0))
for muAna in [globalAnalyzerTnP.binParams, 
              globalAnalyzerRef.binParams]:
    for e in [pA_2016, ppRef_2017, pp_on_AA]:
	    e.toModify(muAna, ptCoarse = cms.untracked.vdouble(0.,1.,2.,3.,4.,5.,7.,9.,12.,15.,20.,30.,40.))
