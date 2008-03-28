import FWCore.ParameterSet.Config as cms

# Select Reco
from L1TriggerOffline.L1Analyzer.RecoSelection_cff import *
# Select L1
from L1TriggerOffline.L1Analyzer.L1Selection_cff import *
# Histogram limits
from L1TriggerOffline.L1Analyzer.HistoLimits_cfi import *
# Root output file
from L1TriggerOffline.L1Analyzer.TFile_cfi import *
# Match Reco and L1 Met 
MatchMetReco = cms.EDFilter("TrivialDeltaRMatcher",
    src = cms.InputTag("SelectL1Met"),
    distMin = cms.double(0.5),
    matched = cms.InputTag("SelectRecoMet")
)

# Match L1 and Reco Met
MatchRecoMet = cms.EDFilter("TrivialDeltaRMatcher",
    src = cms.InputTag("SelectRecoMet"),
    distMin = cms.double(0.5),
    matched = cms.InputTag("SelectL1Met")
)

# Analyzer
L1AnalyzerMetReco = cms.EDAnalyzer("L1Analyzer",
    histoLimits,
    EffMatchMapSource = cms.untracked.InputTag("MatchRecoMet"),
    ReferenceSource = cms.untracked.InputTag("SelectRecoMet"),
    CandidateSource = cms.untracked.InputTag("SelectL1Met"),
    ResMatchMapSource = cms.untracked.InputTag("MatchMetReco")
)

# Define analysis sequence
L1MetRecoAnalysis = cms.Sequence(RecoMetSelection+L1MetSelection*MatchMetReco+MatchRecoMet*L1AnalyzerMetReco)

