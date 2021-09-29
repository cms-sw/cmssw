import FWCore.ParameterSet.Config as cms

MTDTimeCalibESProducer = cms.ESProducer("MTDTimeCalibESProducer",
    BTLLightCollSlope = cms.double(0.075),
    BTLLightCollTime = cms.double(0.2),
    BTLTimeOffset = cms.double(0.0115),
    ETLTimeOffset = cms.double(0.0066),
    appendToDataLabel = cms.string('')
)
