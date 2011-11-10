import FWCore.ParameterSet.Config as cms

l1tGt = cms.EDAnalyzer("L1TGT",
    DQMStore = cms.untracked.bool(True),
    gtEvmSource = cms.InputTag("gtEvmDigis"),
    gtSource = cms.InputTag("gtDigis"),
    disableROOToutput = cms.untracked.bool(True),
    verbose = cms.untracked.bool(False)
)


