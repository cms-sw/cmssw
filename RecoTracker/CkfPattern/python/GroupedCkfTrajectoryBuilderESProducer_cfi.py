import FWCore.ParameterSet.Config as cms

GroupedCkfTrajectoryBuilder = cms.ESProducer("GroupedCkfTrajectoryBuilderESProducer",
    bestHitOnly = cms.bool(True),
    propagatorAlong = cms.string('PropagatorWithMaterial'),
#    propagatorAlong = cms.string('PropagatorWithMaterialParabolicMf'),
    # Filter used on tracks at end of all tracking (in-out + out-in)
    trajectoryFilterName = cms.string('ckfBaseTrajectoryFilter'),
    # Filter used on tracks at end of in-out tracking phase
    inOutTrajectoryFilterName = cms.string('ckfBaseTrajectoryFilter'),
#    inOutTrajectoryFilterName = cms.string('ckfBaseInOutTrajectoryFilter'),
    # If true, then the inOutTrajectoryFilterName will be ignored
    # and the trajectoryFilterName will be used for in-out tracking too.
    useSameTrajFilter = cms.bool(True),
    # Maximum number of track candidates followed at each step of
    # track building
    maxCand = cms.int32(5),
    ComponentName = cms.string('GroupedCkfTrajectoryBuilder'),
    intermediateCleaning = cms.bool(True),
    # Chi2 added to track candidate if no hit found in layer
    lostHitPenalty = cms.double(30.0),
    MeasurementTrackerName = cms.string(''),
    lockHits = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    foundHitBonus = cms.double(5.0),
    updator = cms.string('KFUpdator'),
    # If true, track building will allow for possibility of no hit
    # in a given layer, even if it finds compatible hits there.
    alwaysUseInvalidHits = cms.bool(True),
    requireSeedHitsInRebuild = cms.bool(True),
    keepOriginalIfRebuildFails = cms.bool(False),
    estimator = cms.string('Chi2'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
#    propagatorOpposite = cms.string('PropagatorWithMaterialParabolicMfOpposite'),
    # Out-in tracking will not be attempted unless this many hits
    # are on track after in-out tracking phase.
    minNrOfHitsForRebuild = cms.int32(5)
)


