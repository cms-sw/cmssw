import FWCore.ParameterSet.Config as cms

# Select MC truth
from L1TriggerOffline.L1Analyzer.GenSelection_cff import *
# Select L1
from L1TriggerOffline.L1Analyzer.L1Selection_cff import *
# Histogram limits
from L1TriggerOffline.L1Analyzer.HistoLimits_cfi import *
# Root output file
from L1TriggerOffline.L1Analyzer.TFile_cfi import *
# Match generator and L1 Met 
MatchMetMc = cms.EDFilter("TrivialDeltaRMatcher",
    src = cms.InputTag("SelectL1Met"),
    distMin = cms.double(0.5),
    matched = cms.InputTag("SelectGenMet")
)

# Match L1 and generator Met
MatchMcMet = cms.EDFilter("TrivialDeltaRMatcher",
    src = cms.InputTag("SelectGenMet"),
    distMin = cms.double(0.5),
    matched = cms.InputTag("SelectL1Met")
)

# Analyzer
L1AnalyzerMetMC = cms.EDAnalyzer("L1Analyzer",
    histoLimits,
    EffMatchMapSource = cms.untracked.InputTag("MatchMcMet"),
    ReferenceSource = cms.untracked.InputTag("SelectGenMet"),
    CandidateSource = cms.untracked.InputTag("SelectL1Met"),
    ResMatchMapSource = cms.untracked.InputTag("MatchMetMc")
)

# Define analysis sequence
L1MetMCAnalysis = cms.Sequence(CloneL1ExtraMet*CloneGenMet*SelectL1Met*SelectGenMet*MatchMetMc*MatchMcMet*L1AnalyzerMetMC)

