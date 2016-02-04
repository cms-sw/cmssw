import FWCore.ParameterSet.Config as cms

# Select Reco
from L1TriggerOffline.L1Analyzer.RecoSelection_cff import *
# Select L1
from L1TriggerOffline.L1Analyzer.L1Selection_cff import *
# Histogram limits
from L1TriggerOffline.L1Analyzer.HistoLimits_cfi import *
# Root output file
from L1TriggerOffline.L1Analyzer.TFile_cfi import *
# Match Reco and L1 jets 
MatchCenJetsReco = cms.EDFilter("TrivialDeltaRMatcher",
    src = cms.InputTag("SelectL1CenJets"),
    distMin = cms.double(0.5),
    matched = cms.InputTag("SelectRecoCenJets")
)

# Match L1 and Reco jets
MatchRecoCenJets = cms.EDFilter("TrivialDeltaRMatcher",
    src = cms.InputTag("SelectRecoCenJets"),
    distMin = cms.double(0.5),
    matched = cms.InputTag("SelectL1CenJets")
)

# Analyzer
L1AnalyzerCenJetsReco = cms.EDAnalyzer("L1Analyzer",
    histoLimits,
    EffMatchMapSource = cms.untracked.InputTag("MatchRecoCenJets"),
    ReferenceSource = cms.untracked.InputTag("SelectRecoCenJets"),
    CandidateSource = cms.untracked.InputTag("SelectL1CenJets"),
    ResMatchMapSource = cms.untracked.InputTag("MatchCenJetsReco")
)

# Define analysis sequence
L1CenJetRecoAnalysis = cms.Sequence(RecoCenJetSelection+L1CenJetSelection*MatchCenJetsReco+MatchRecoCenJets*L1AnalyzerCenJetsReco)

