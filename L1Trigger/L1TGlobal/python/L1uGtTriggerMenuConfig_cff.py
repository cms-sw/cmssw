import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TGlobal.l1uGtTriggerMenuXml_cfi import *
#
L1uGtTriggerMenuRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1uGtTriggerMenuRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


