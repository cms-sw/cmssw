import FWCore.ParameterSet.Config as cms

from L1TriggerConfig.L1GtConfigProducers.l1GtBoardMaps_cfi import *
#
L1GtBoardMapsRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GtBoardMapsRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


