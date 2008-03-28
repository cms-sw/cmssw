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
MatchTauJetsReco = cms.EDFilter("TrivialDeltaRMatcher",
    src = cms.InputTag("SelectL1TauJets"),
    distMin = cms.double(0.5),
    matched = cms.InputTag("SelectRecoTauJets")
)

# Match L1 and Reco tau jets
MatchRecoTauJets = cms.EDFilter("TrivialDeltaRMatcher",
    src = cms.InputTag("SelectRecoTauJets"),
    distMin = cms.double(0.5),
    matched = cms.InputTag("SelectL1TauJets")
)

# Analyzer
L1AnalyzerTauJetsReco = cms.EDAnalyzer("L1Analyzer",
    histoLimits,
    EffMatchMapSource = cms.untracked.InputTag("MatchRecoTauJets"),
    ReferenceSource = cms.untracked.InputTag("SelectRecoTauJets"),
    CandidateSource = cms.untracked.InputTag("SelectL1TauJets"),
    ResMatchMapSource = cms.untracked.InputTag("MatchTauJetsReco")
)

# Define analysis sequence
L1TauJetRecoAnalysis = cms.Sequence(RecoTauJetSelection+L1TauJetSelection*MatchTauJetsReco+MatchRecoTauJets*L1AnalyzerTauJetsReco)

