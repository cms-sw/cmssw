import FWCore.ParameterSet.Config as cms

# Select Reco
from L1TriggerOffline.L1Analyzer.RecoSelection_cff import *
# Select L1
from L1TriggerOffline.L1Analyzer.L1Selection_cff import *
# Histogram limits
from L1TriggerOffline.L1Analyzer.HistoLimits_cfi import *
# Root output file
from L1TriggerOffline.L1Analyzer.TFile_cfi import *
# Match Reco and L1 electrons
MatchNonIsoEmReco = cms.EDFilter("TrivialDeltaRMatcher",
    src = cms.InputTag("SelectL1NonIsoEm"),
    distMin = cms.double(0.5),
    matched = cms.InputTag("SelectRecoElec")
)

# Match L1 and Reco electrons
MatchRecoNonIsoEm = cms.EDFilter("TrivialDeltaRMatcher",
    src = cms.InputTag("SelectRecoElec"),
    distMin = cms.double(0.5),
    matched = cms.InputTag("SelectL1NonIsoEm")
)

# Analyzer
L1AnalyzerNonIsoEmReco = cms.EDAnalyzer("L1Analyzer",
    histoLimits,
    EffMatchMapSource = cms.untracked.InputTag("MatchRecoNonIsoEm"),
    ReferenceSource = cms.untracked.InputTag("SelectRecoElec"),
    CandidateSource = cms.untracked.InputTag("SelectL1NonIsoEm"),
    ResMatchMapSource = cms.untracked.InputTag("MatchNonIsoEmReco")
)

# Define analysis sequence
L1NonIsoEmRecoAnalysis = cms.Sequence(RecoElecSelection+L1NonIsoEmSelection*MatchNonIsoEmReco+MatchRecoNonIsoEm*L1AnalyzerNonIsoEmReco)

