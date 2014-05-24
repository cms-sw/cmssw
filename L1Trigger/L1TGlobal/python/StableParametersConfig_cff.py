import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TGlobal.StableParameters_cfi import *
#
StableParametersRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1TGlobalStableParametersRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


