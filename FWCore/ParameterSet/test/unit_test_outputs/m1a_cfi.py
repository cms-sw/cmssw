import FWCore.ParameterSet.Config as cms

m1a = cms.EDProducer("IntProducer",
    ivalue = cms.int32(2)
)
