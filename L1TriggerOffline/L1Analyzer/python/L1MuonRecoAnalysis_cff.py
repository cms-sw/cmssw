import FWCore.ParameterSet.Config as cms

# Select Reco
from L1TriggerOffline.L1Analyzer.RecoSelection_cff import *
# Select L1
from L1TriggerOffline.L1Analyzer.L1Selection_cff import *
# Histogram limits
from L1TriggerOffline.L1Analyzer.HistoLimits_cfi import *
# Root output file
from L1TriggerOffline.L1Analyzer.TFile_cfi import *
# Match Reco and L1 muons 
MatchMuonsReco = cms.EDFilter("TrivialDeltaRMatcher",
    src = cms.InputTag("SelectL1Muons"),
    distMin = cms.double(0.5),
    matched = cms.InputTag("SelectRecoMuons")
)

# Match L1 and Reco muons
MatchRecoMuons = cms.EDFilter("TrivialDeltaRMatcher",
    src = cms.InputTag("SelectRecoMuons"),
    distMin = cms.double(0.5),
    matched = cms.InputTag("SelectL1Muons")
)

# Analyzer
L1AnalyzerMuonReco = cms.EDAnalyzer("L1Analyzer",
    histoLimits,
    EffMatchMapSource = cms.untracked.InputTag("MatchRecoMuons"),
    ReferenceSource = cms.untracked.InputTag("SelectRecoMuons"),
    CandidateSource = cms.untracked.InputTag("SelectL1Muons"),
    ResMatchMapSource = cms.untracked.InputTag("MatchMuonsReco")
)

# Define analysis sequence
L1MuonRecoAnalysis = cms.Sequence(RecoMuonSelection+L1MuonSelection*MatchMuonsReco+MatchRecoMuons*L1AnalyzerMuonReco)

