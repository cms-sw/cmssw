import FWCore.ParameterSet.Config as cms

# Select MC truth
from L1TriggerOffline.L1Analyzer.GenSelection_cff import *
# Select L1
from L1TriggerOffline.L1Analyzer.L1Selection_cff import *
# Histogram limits
from L1TriggerOffline.L1Analyzer.HistoLimits_cfi import *
# Root output file
from L1TriggerOffline.L1Analyzer.TFile_cfi import *
# Match generator and L1 muons 
MatchMuonsMc = cms.EDFilter("TrivialDeltaRMatcher",
    src = cms.InputTag("SelectL1Muons"),
    distMin = cms.double(0.5),
    matched = cms.InputTag("SelectGenMuons")
)

# Match L1 and generator muons
MatchMcMuons = cms.EDFilter("TrivialDeltaRMatcher",
    src = cms.InputTag("SelectGenMuons"),
    distMin = cms.double(0.5),
    matched = cms.InputTag("SelectL1Muons")
)

# Analyzer
L1AnalyzerMuonMC = cms.EDAnalyzer("L1Analyzer",
    histoLimits,
    EffMatchMapSource = cms.untracked.InputTag("MatchMcMuons"),
    ReferenceSource = cms.untracked.InputTag("SelectGenMuons"),
    CandidateSource = cms.untracked.InputTag("SelectL1Muons"),
    ResMatchMapSource = cms.untracked.InputTag("MatchMuonsMc")
)

# Define analysis sequence
L1MuonMCAnalysis = cms.Sequence(L1MuonSelection+GenMuonSelection*MatchMuonsMc+MatchMcMuons*L1AnalyzerMuonMC)

