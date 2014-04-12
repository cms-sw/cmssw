import FWCore.ParameterSet.Config as cms

from L1TriggerConfig.L1GtConfigProducers.l1GtPrescaleFactorsAlgoTrig_cfi import *
#
L1GtPrescaleFactorsAlgoTrigRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GtPrescaleFactorsAlgoTrigRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


