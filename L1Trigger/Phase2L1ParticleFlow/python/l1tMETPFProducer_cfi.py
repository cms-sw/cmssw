import FWCore.ParameterSet.Config as cms

l1tMETPFProducer = cms.EDProducer("L1MetPfProducer",
                                 L1PFObjects = cms.InputTag("l1tLayer1","Puppi"),
                                 maxCands = cms.int32(128),
                                 modelVersion = cms.string(""),
)

l1tMETMLProducer = cms.EDProducer("L1MetPfProducer",
                                 L1PFObjects = cms.InputTag("l1tLayer1","Puppi"),
                                 maxCands = cms.int32(100),
                                 modelVersion = cms.string("L1METML_v1"),
)
