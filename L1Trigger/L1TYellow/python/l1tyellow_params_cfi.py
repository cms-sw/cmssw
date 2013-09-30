import FWCore.ParameterSet.Config as cms

yellowParams = cms.ESSource(
    "L1TYellowParamsESProducer",
    firmwareVersion = cms.uint32(1),
    paramA = cms.uint32(2),
    paramB = cms.uint32(1),
    paramC = cms.uint32(3),
    label = cms.string("L1TYellowParamsWriter")
    )
