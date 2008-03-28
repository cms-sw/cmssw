import FWCore.ParameterSet.Config as cms

# Select MC truth
from L1TriggerOffline.L1Analyzer.GenSelection_cff import *
# Select L1
from L1TriggerOffline.L1Analyzer.L1Selection_cff import *
# Histogram limits
from L1TriggerOffline.L1Analyzer.HistoLimits_cfi import *
# Root output file
from L1TriggerOffline.L1Analyzer.TFile_cfi import *
# Match generator taus and L1 tau jets 
MatchTauJetsMc = cms.EDFilter("TrivialDeltaRMatcher",
    src = cms.InputTag("SelectL1TauJets"),
    distMin = cms.double(0.5),
    matched = cms.InputTag("SelectGenTauJets")
)

# Match L1 and generator tau jets
MatchMcTauJets = cms.EDFilter("TrivialDeltaRMatcher",
    src = cms.InputTag("SelectGenTauJets"),
    distMin = cms.double(0.5),
    matched = cms.InputTag("SelectL1TauJets")
)

# Analyzer
L1AnalyzerTauJetsMC = cms.EDAnalyzer("L1Analyzer",
    histoLimits,
    EffMatchMapSource = cms.untracked.InputTag("MatchMcTauJets"),
    ReferenceSource = cms.untracked.InputTag("SelectGenTauJets"),
    CandidateSource = cms.untracked.InputTag("SelectL1TauJets"),
    ResMatchMapSource = cms.untracked.InputTag("MatchTauJetsMc")
)

# Define analysis sequence
L1TauJetMCAnalysis = cms.Sequence(L1TauJetSelection+GenTauJetSelection*MatchTauJetsMc+MatchMcTauJets*L1AnalyzerTauJetsMC)

