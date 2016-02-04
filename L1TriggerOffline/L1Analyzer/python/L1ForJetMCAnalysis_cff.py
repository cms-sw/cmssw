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
MatchForJetsMc = cms.EDFilter("TrivialDeltaRMatcher",
    src = cms.InputTag("SelectL1ForJets"),
    distMin = cms.double(0.5),
    matched = cms.InputTag("SelectGenForJets")
)

# Match L1 and generator jets
MatchMcForJets = cms.EDFilter("TrivialDeltaRMatcher",
    src = cms.InputTag("SelectGenForJets"),
    distMin = cms.double(0.5),
    matched = cms.InputTag("SelectL1ForJets")
)

# Analyzer
L1AnalyzerForJetsMC = cms.EDAnalyzer("L1Analyzer",
    histoLimits,
    EffMatchMapSource = cms.untracked.InputTag("MatchMcForJets"),
    ReferenceSource = cms.untracked.InputTag("SelectGenForJets"),
    CandidateSource = cms.untracked.InputTag("SelectL1ForJets"),
    ResMatchMapSource = cms.untracked.InputTag("MatchForJetsMc")
)

# Define analysis sequence
L1ForJetMCAnalysis = cms.Sequence(L1ForJetSelection+GenForJetSelection*MatchForJetsMc+MatchMcForJets*L1AnalyzerForJetsMC)

