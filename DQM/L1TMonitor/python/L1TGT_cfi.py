import FWCore.ParameterSet.Config as cms

l1tGt = cms.EDAnalyzer("L1TGT",
    gtEvmSource = cms.InputTag("gtEvmDigis"),
    gtSource = cms.InputTag("gtDigis"),
    verbose = cms.untracked.bool(False)
)


