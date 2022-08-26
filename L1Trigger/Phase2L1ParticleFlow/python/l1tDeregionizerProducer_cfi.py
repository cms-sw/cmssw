import FWCore.ParameterSet.Config as cms

l1tDeregionizerProducer = cms.EDProducer("DeregionizerProducer",
                           RegionalPuppiCands  = cms.InputTag("l1tLayer1","PuppiRegional"),
                           nPuppiFinalBuffer   = cms.uint32(128),
                           nPuppiPerClk        = cms.uint32(6),
                           nPuppiFirstBuffers  = cms.uint32(12),
                           nPuppiSecondBuffers = cms.uint32(32),
                           nPuppiThirdBuffers  = cms.uint32(64)
                         )

l1tDeregionizerProducerExtended = l1tDeregionizerProducer.clone(RegionalPuppiCands  = cms.InputTag("l1tLayer1Extended","PuppiRegional"))
