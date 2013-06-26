import FWCore.ParameterSet.Config as cms

filterTrackerOn = cms.EDFilter(
    'FilterTrackerOn',
    MinModulesWithHVoff = cms.int32(500)
)
