import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TGlobal.l1uGtStableParameters_cfi import *
#
L1uGtStableParametersRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1uGtStableParametersRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


