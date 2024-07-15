import FWCore.ParameterSet.Config as cms

l1AXOTree = cms.EDAnalyzer("L1AXOTreeProducer",
                           axoscoreToken    = cms.untracked.InputTag("simGtStage2Digis","AXOScore") 
)
