import FWCore.ParameterSet.Config as cms

# Select Reco
from L1TriggerOffline.L1Analyzer.RecoSelection_cff import *
# Select L1
from L1TriggerOffline.L1Analyzer.L1Selection_cff import *
# Histogram limits
from L1TriggerOffline.L1Analyzer.HistoLimits_cfi import *
# Root output file
from L1TriggerOffline.L1Analyzer.TFile_cfi import *
# Match Reco and L1 electrons
MatchIsoEmReco = cms.EDFilter("TrivialDeltaRMatcher",
    src = cms.InputTag("SelectL1IsoEm"),
    distMin = cms.double(0.5),
    matched = cms.InputTag("SelectRecoElec")
)

# Match L1 and Reco electrons
MatchRecoIsoEm = cms.EDFilter("TrivialDeltaRMatcher",
    src = cms.InputTag("SelectRecoElec"),
    distMin = cms.double(0.5),
    matched = cms.InputTag("SelectL1IsoEm")
)

# Analyzer
L1AnalyzerIsoEmReco = cms.EDAnalyzer("L1Analyzer",
    histoLimits,
    EffMatchMapSource = cms.untracked.InputTag("MatchRecoIsoEm"),
    ReferenceSource = cms.untracked.InputTag("SelectRecoElec"),
    CandidateSource = cms.untracked.InputTag("SelectL1IsoEm"),
    ResMatchMapSource = cms.untracked.InputTag("MatchIsoEmReco")
)

# Define analysis sequence
L1IsoEmRecoAnalysis = cms.Sequence(RecoElecSelection+L1IsoEmSelection*MatchIsoEmReco+MatchRecoIsoEm*L1AnalyzerIsoEmReco)

