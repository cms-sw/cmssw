import FWCore.ParameterSet.Config as cms

trajectoryCleanerForShortTracks = cms.ESProducer("TrajectoryCleanerESProducer",
    ComponentName = cms.string('TrajectoryCleanerForShortTracks')
)


