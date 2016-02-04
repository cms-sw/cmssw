import FWCore.ParameterSet.Config as cms

L1TriggerKeyListRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1TriggerKeyListRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


