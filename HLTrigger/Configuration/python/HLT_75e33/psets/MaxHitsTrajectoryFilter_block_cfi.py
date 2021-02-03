import FWCore.ParameterSet.Config as cms

MaxHitsTrajectoryFilter_block = cms.PSet(
    ComponentType = cms.string('MaxHitsTrajectoryFilter'),
    maxNumberOfHits = cms.int32(100)
)