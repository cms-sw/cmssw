import FWCore.ParameterSet.Config as cms

l1MenuTreeProducer = cms.EDAnalyzer("L1MenuTreeProducer",
   L1MenuInputTag = cms.InputTag("l1GtTriggerMenuLite")
)

