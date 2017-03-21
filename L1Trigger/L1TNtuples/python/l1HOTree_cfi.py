import FWCore.ParameterSet.Config as cms

l1HOTree = cms.EDAnalyzer(
    "L1HOTreeProducer",
    hoDataFrameToken = cms.untracked.InputTag("hcalDigis")
)
