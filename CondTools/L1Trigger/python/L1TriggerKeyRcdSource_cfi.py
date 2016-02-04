import FWCore.ParameterSet.Config as cms

L1TriggerKeyRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1TriggerKeyRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


