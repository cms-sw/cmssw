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
MatchCenTauJetsMc = cms.EDFilter("TrivialDeltaRMatcher",
    src = cms.InputTag("SelectL1CenTauJets"),
    distMin = cms.double(0.5),
    matched = cms.InputTag("SelectGenCenJets")
)

# Match L1 and generator jets
MatchMcCenTauJets = cms.EDFilter("TrivialDeltaRMatcher",
    src = cms.InputTag("SelectGenCenJets"),
    distMin = cms.double(0.5),
    matched = cms.InputTag("SelectL1CenTauJets")
)

# Analyzer
L1AnalyzerCenTauJetsMC = cms.EDAnalyzer("L1Analyzer",
    histoLimits,
    EffMatchMapSource = cms.untracked.InputTag("MatchMcCenTauJets"),
    ReferenceSource = cms.untracked.InputTag("SelectGenCenJets"),
    CandidateSource = cms.untracked.InputTag("SelectL1CenTauJets"),
    ResMatchMapSource = cms.untracked.InputTag("MatchCenTauJetsMc")
)

# Define analysis sequence
L1CenTauJetMCAnalysis = cms.Sequence(L1CenTauJetSelection
                                     +GenCenJetSelection
                                     *MatchCenTauJetsMc
                                     +MatchMcCenTauJets
                                     *L1AnalyzerCenTauJetsMC)

