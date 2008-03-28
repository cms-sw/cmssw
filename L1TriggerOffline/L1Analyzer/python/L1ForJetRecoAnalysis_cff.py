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
MatchForJetsReco = cms.EDFilter("TrivialDeltaRMatcher",
    src = cms.InputTag("SelectL1ForJets"),
    distMin = cms.double(0.5),
    matched = cms.InputTag("SelectRecoForJets")
)

# Match L1 and Reco jets
MatchRecoForJets = cms.EDFilter("TrivialDeltaRMatcher",
    src = cms.InputTag("SelectRecoForJets"),
    distMin = cms.double(0.5),
    matched = cms.InputTag("SelectL1ForJets")
)

# Analyzer
L1AnalyzerForJetsReco = cms.EDAnalyzer("L1Analyzer",
    histoLimits,
    EffMatchMapSource = cms.untracked.InputTag("MatchRecoForJets"),
    ReferenceSource = cms.untracked.InputTag("SelectRecoForJets"),
    CandidateSource = cms.untracked.InputTag("SelectL1ForJets"),
    ResMatchMapSource = cms.untracked.InputTag("MatchForJetsReco")
)

# Define analysis sequence
L1ForJetRecoAnalysis = cms.Sequence(RecoForJetSelection+L1ForJetSelection*MatchForJetsReco+MatchRecoForJets*L1AnalyzerForJetsReco)

