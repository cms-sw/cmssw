import FWCore.ParameterSet.Config as cms

from L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMenuXml_cfi import *
#
L1GtTriggerMenuRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GtTriggerMenuRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


