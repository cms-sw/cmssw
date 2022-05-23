import FWCore.ParameterSet.Config as cms

L1MetPfProducer = cms.EDProducer("L1MetPfProducer",
                                 L1PFObjects = cms.InputTag("l1ctLayer1","Puppi"),
                                 maxCands = cms.int32(128),
)
