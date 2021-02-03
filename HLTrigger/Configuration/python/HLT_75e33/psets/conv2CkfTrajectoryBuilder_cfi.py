import FWCore.ParameterSet.Config as cms

conv2CkfTrajectoryBuilder = cms.PSet(
    ComponentType = cms.string('GroupedCkfTrajectoryBuilder'),
    MeasurementTrackerName = cms.string(''),
    TTRHBuilder = cms.string('WithTrackAngle'),
    alwaysUseInvalidHits = cms.bool(True),
    bestHitOnly = cms.bool(True),
    clustersToSkip = cms.InputTag("conv2Clusters"),
    estimator = cms.string('Chi2'),
    foundHitBonus = cms.double(10.0),
    inOutTrajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('CkfBaseTrajectoryFilter_block')
    ),
    intermediateCleaning = cms.bool(True),
    keepOriginalIfRebuildFails = cms.bool(False),
    lockHits = cms.bool(True),
    lostHitPenalty = cms.double(30.0),
    maxCand = cms.int32(2),
    minNrOfHitsForRebuild = cms.int32(3),
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    requireSeedHitsInRebuild = cms.bool(True),
    seedAs5DHit = cms.bool(False),
    trajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('conv2CkfTrajectoryFilter')
    ),
    updator = cms.string('KFUpdator'),
    useSameTrajFilter = cms.bool(True)
)