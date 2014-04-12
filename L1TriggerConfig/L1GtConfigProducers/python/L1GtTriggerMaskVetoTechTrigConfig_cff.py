import FWCore.ParameterSet.Config as cms

from L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMaskVetoTechTrig_cfi import *
#
L1GtTriggerMaskVetoTechTrigRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GtTriggerMaskVetoTechTrigRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


