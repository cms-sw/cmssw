import FWCore.ParameterSet.Config as cms

from L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMaskAlgoTrig_cfi import *
#
L1GtTriggerMaskAlgoTrigRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GtTriggerMaskAlgoTrigRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


