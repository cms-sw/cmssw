import FWCore.ParameterSet.Config as cms

from L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMaskTechTrig_cfi import *
#
L1GtTriggerMaskTechTrigRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GtTriggerMaskTechTrigRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


