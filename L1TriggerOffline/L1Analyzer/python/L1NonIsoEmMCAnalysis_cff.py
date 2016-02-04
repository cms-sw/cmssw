import FWCore.ParameterSet.Config as cms

# Select MC truth
from L1TriggerOffline.L1Analyzer.GenSelection_cff import *
# Select L1
from L1TriggerOffline.L1Analyzer.L1Selection_cff import *
# Histogram limits
from L1TriggerOffline.L1Analyzer.HistoLimits_cfi import *
# Root output file
from L1TriggerOffline.L1Analyzer.TFile_cfi import *
# Match generator and L1 electrons
MatchNonIsoEmMc = cms.EDFilter("TrivialDeltaRMatcher",
    src = cms.InputTag("SelectL1NonIsoEm"),
    distMin = cms.double(0.5),
    matched = cms.InputTag("SelectGenElec")
)

# Match L1 and generator electrons
MatchMcNonIsoEm = cms.EDFilter("TrivialDeltaRMatcher",
    src = cms.InputTag("SelectGenElec"),
    distMin = cms.double(0.5),
    matched = cms.InputTag("SelectL1NonIsoEm")
)

# Analyzer
L1AnalyzerNonIsoEmMC = cms.EDAnalyzer("L1Analyzer",
    histoLimits,
    EffMatchMapSource = cms.untracked.InputTag("MatchMcNonIsoEm"),
    ReferenceSource = cms.untracked.InputTag("SelectGenElec"),
    CandidateSource = cms.untracked.InputTag("SelectL1NonIsoEm"),
    ResMatchMapSource = cms.untracked.InputTag("MatchNonIsoEmMc")
)

# Define analysis sequence
L1NonIsoEmMCAnalysis = cms.Sequence(L1NonIsoEmSelection+GenElecSelection*MatchNonIsoEmMc+MatchMcNonIsoEm*L1AnalyzerNonIsoEmMC)

