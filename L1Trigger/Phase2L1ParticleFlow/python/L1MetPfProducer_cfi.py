import FWCore.ParameterSet.Config as cms

L1MetPfProducer = cms.EDProducer("L1MetPfProducer",
                                 L1PFObjects = cms.InputTag("L1PFProducer","l1pfCandidates"),
                                 sinCosTableBits = cms.uint32(8),
                                 inverseDropBits = cms.uint32(2),
                                 arcTanTableBits = cms.uint32(7)
                             )
