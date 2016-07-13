import FWCore.ParameterSet.Config as cms

L1TriggerKeyRcdSourceExt = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1TriggerKeyExtRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


