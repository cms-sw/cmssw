import FWCore.ParameterSet.Config as cms

# Select Reco
from L1TriggerOffline.L1Analyzer.RecoSelection_cff import *
# Select L1
from L1TriggerOffline.L1Analyzer.L1Selection_cff import *
# Histogram limits
from L1TriggerOffline.L1Analyzer.HistoLimits_cfi import *
# Root output file
from L1TriggerOffline.L1Analyzer.TFile_cfi import *

# Match Reco and L1 tau jets 
MatchMergedJetsReco = cms.EDFilter("TrivialDeltaRMatcher",
    src = cms.InputTag("SelectMergedL1ExtraJets"),
    distMin = cms.double(1.0),
    matched = cms.InputTag("SelectRecoCenJets")                              
)

# Match L1 and Reco tau jets
MatchRecoMergedJets = cms.EDFilter("TrivialDeltaRMatcher",
    src = cms.InputTag("SelectRecoCenJets"),                       
    distMin = cms.double(1.0),
    matched = cms.InputTag("SelectMergedL1ExtraJets")
)

# Analyzer
L1AnalyzerMergedJetsReco = cms.EDAnalyzer("L1Analyzer",
    histoLimits,
    EffMatchMapSource = cms.untracked.InputTag("MatchRecoMergedJets"),
    ReferenceSource = cms.untracked.InputTag("SelectRecoCenJets"),                
    CandidateSource = cms.untracked.InputTag("SelectMergedL1ExtraJets"),
    ResMatchMapSource = cms.untracked.InputTag("MatchMergedJetsReco")
)

# Define analysis sequence
L1MergedJetRecoAnalysis = cms.Sequence(RecoCenJetSelection+L1MergedJetSelection*MatchMergedJetsReco+MatchRecoMergedJets*L1AnalyzerMergedJetsReco)
