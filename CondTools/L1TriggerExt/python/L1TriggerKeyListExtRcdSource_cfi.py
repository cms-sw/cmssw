import FWCore.ParameterSet.Config as cms

L1TriggerKeyListExtRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1TriggerKeyListExtRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


