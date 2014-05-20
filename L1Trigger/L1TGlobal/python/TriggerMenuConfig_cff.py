import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TGlobal.TriggerMenuXml_cfi import *
#
TriggerMenuRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1TGlobalTriggerMenuRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


