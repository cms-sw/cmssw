import FWCore.ParameterSet.Config as cms

hltL1GTAcceptFilter = cms.EDFilter("L1GTAcceptFilter",
                                   algoBlocksTag = cms.InputTag("l1tGTAlgoBlockProducer"),
                                   decision = cms.string("final")                                    
                                   )
