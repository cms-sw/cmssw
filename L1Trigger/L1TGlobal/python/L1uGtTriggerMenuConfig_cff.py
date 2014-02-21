import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TGlobal.l1uGtTriggerMenuXml_cfi import *
#
L1GtTriggerMenuRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GtTriggerMenuRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


