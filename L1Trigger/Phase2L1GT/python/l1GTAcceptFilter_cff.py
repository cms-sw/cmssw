import FWCore.ParameterSet.Config as cms

l1tGTAcceptFilter = cms.EDFilter(
    "L1GTAcceptFilter",
    algoBlocksTag = cms.InputTag("l1tGTAlgoBlockProducer"),
)
