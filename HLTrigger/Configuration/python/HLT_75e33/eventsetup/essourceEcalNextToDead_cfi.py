import FWCore.ParameterSet.Config as cms

essourceEcalNextToDead = cms.ESSource("EmptyESSource",
    firstValid = cms.vuint32(1),
    iovIsRunNotTime = cms.bool(True),
    recordName = cms.string('EcalNextToDeadChannelRcd')
)
