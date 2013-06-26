import FWCore.ParameterSet.Config as cms

from L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMaskVetoAlgoTrig_cfi import *
#
L1GtTriggerMaskVetoAlgoTrigRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GtTriggerMaskVetoAlgoTrigRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


