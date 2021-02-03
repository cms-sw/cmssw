import FWCore.ParameterSet.Config as cms

initialStepTrajectoryBuilderPreSplitting = cms.PSet(
    ComponentType = cms.string('GroupedCkfTrajectoryBuilder'),
    MeasurementTrackerName = cms.string(''),
    TTRHBuilder = cms.string('WithTrackAngle'),
    alwaysUseInvalidHits = cms.bool(True),
    bestHitOnly = cms.bool(True),
    estimator = cms.string('initialStepChi2Est'),
    foundHitBonus = cms.double(10.0),
    inOutTrajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('CkfBaseTrajectoryFilter_block')
    ),
    intermediateCleaning = cms.bool(True),
    keepOriginalIfRebuildFails = cms.bool(False),
    lockHits = cms.bool(True),
    lostHitPenalty = cms.double(30.0),
    maxCand = cms.int32(3),
    minNrOfHitsForRebuild = cms.int32(5),
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    requireSeedHitsInRebuild = cms.bool(True),
    seedAs5DHit = cms.bool(False),
    trajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('initialStepTrajectoryFilterPreSplitting')
    ),
    updator = cms.string('KFUpdator'),
    useSameTrajFilter = cms.bool(True)
)