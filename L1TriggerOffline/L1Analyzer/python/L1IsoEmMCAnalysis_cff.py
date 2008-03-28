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
MatchIsoEmMc = cms.EDFilter("TrivialDeltaRMatcher",
    src = cms.InputTag("SelectL1IsoEm"),
    distMin = cms.double(0.5),
    matched = cms.InputTag("SelectGenElec")
)

# Match L1 and generator electrons
MatchMcIsoEm = cms.EDFilter("TrivialDeltaRMatcher",
    src = cms.InputTag("SelectGenElec"),
    distMin = cms.double(0.5),
    matched = cms.InputTag("SelectL1IsoEm")
)

# Analyzer
L1AnalyzerIsoEmMC = cms.EDAnalyzer("L1Analyzer",
    histoLimits,
    EffMatchMapSource = cms.untracked.InputTag("MatchMcIsoEm"),
    ReferenceSource = cms.untracked.InputTag("SelectGenElec"),
    CandidateSource = cms.untracked.InputTag("SelectL1IsoEm"),
    ResMatchMapSource = cms.untracked.InputTag("MatchIsoEmMc")
)

# Define analysis sequence
L1IsoEmMCAnalysis = cms.Sequence(L1IsoEmSelection+GenElecSelection*MatchIsoEmMc+MatchMcIsoEm*L1AnalyzerIsoEmMC)

