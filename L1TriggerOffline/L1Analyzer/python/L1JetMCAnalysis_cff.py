import FWCore.ParameterSet.Config as cms

# Select MC truth
from L1TriggerOffline.L1Analyzer.GenSelection_cff import *
# Select L1
from L1TriggerOffline.L1Analyzer.L1Selection_cff import *
# Histogram limits
from L1TriggerOffline.L1Analyzer.HistoLimits_cfi import *
# Root output file
from L1TriggerOffline.L1Analyzer.TFile_cfi import *
# Match generator and L1 jets 
MatchJetsMc = cms.EDFilter("TrivialDeltaRMatcher",
    src = cms.InputTag("SelectL1Jets"),
    distMin = cms.double(0.5),
    matched = cms.InputTag("SelectGenJets")
)

# Match L1 and generator jets
MatchMcJets = cms.EDFilter("TrivialDeltaRMatcher",
    src = cms.InputTag("SelectGenJets"),
    distMin = cms.double(0.5),
    matched = cms.InputTag("SelectL1Jets")
)

# Analyzer
L1AnalyzerJetsMC = cms.EDAnalyzer("L1Analyzer",
    histoLimits,
    EffMatchMapSource = cms.untracked.InputTag("MatchMcJets"),
    ReferenceSource = cms.untracked.InputTag("SelectGenJets"),
    CandidateSource = cms.untracked.InputTag("SelectL1Jets"),
    ResMatchMapSource = cms.untracked.InputTag("MatchJetsMc")
)

# Define analysis sequence
L1JetMCAnalysis = cms.Sequence(L1JetSelection+GenJetSelection*MatchJetsMc+MatchMcJets*L1AnalyzerJetsMC)
