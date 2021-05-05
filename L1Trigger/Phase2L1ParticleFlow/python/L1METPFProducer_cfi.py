import FWCore.ParameterSet.Config as cms

L1METPFProducer = cms.EDProducer("L1METPFProducer",
                                 L1PFObjects = cms.InputTag("L1PFProducer","l1pfCandidates"),
                                 maxCandidates = cms.int32(128),
)
