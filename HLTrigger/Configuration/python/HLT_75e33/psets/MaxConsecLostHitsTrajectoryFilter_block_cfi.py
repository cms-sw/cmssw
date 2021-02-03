import FWCore.ParameterSet.Config as cms

MaxConsecLostHitsTrajectoryFilter_block = cms.PSet(
    ComponentType = cms.string('MaxConsecLostHitsTrajectoryFilter'),
    maxConsecLostHits = cms.int32(1)
)