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
MatchEmMc = cms.EDFilter("TrivialDeltaRMatcher",
    src = cms.InputTag("SelectL1Em"),
    distMin = cms.double(0.5),
    matched = cms.InputTag("SelectGenElec")
)

# Match L1 and generator electrons
MatchMcEm = cms.EDFilter("TrivialDeltaRMatcher",
    src = cms.InputTag("SelectGenElec"),
    distMin = cms.double(0.5),
    matched = cms.InputTag("SelectL1Em")
)

# Analyzer
L1AnalyzerEmMC = cms.EDAnalyzer("L1Analyzer",
    histoLimits,
    EffMatchMapSource = cms.untracked.InputTag("MatchMcEm"),
    ReferenceSource = cms.untracked.InputTag("SelectGenElec"),
    CandidateSource = cms.untracked.InputTag("SelectL1Em"),
    ResMatchMapSource = cms.untracked.InputTag("MatchEmMc")
)

# Define analysis sequence
L1EmMCAnalysis = cms.Sequence(L1EmSelection+GenElecSelection*MatchEmMc+MatchMcEm*L1AnalyzerEmMC)

