import FWCore.ParameterSet.Config as cms

yellowParamsSource = cms.ESSource(
    "EmptyESSource",
    recordName = cms.string('L1TYellowParamsRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
    )

yellowParams = cms.ESProducer(
    "l1t::YellowParamsESProducer",
    firmwareVersion = cms.uint32(1),
    paramA = cms.uint32(2),
    paramB = cms.uint32(1),
    paramC = cms.uint32(3),
    label = cms.string("l1t::YellowParamsESProducer")
    )
