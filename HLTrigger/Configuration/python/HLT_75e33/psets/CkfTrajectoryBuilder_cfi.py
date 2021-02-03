import FWCore.ParameterSet.Config as cms

CkfTrajectoryBuilder = cms.PSet(
    ComponentType = cms.string('CkfTrajectoryBuilder'),
    MeasurementTrackerName = cms.string(''),
    TTRHBuilder = cms.string('WithTrackAngle'),
    alwaysUseInvalidHits = cms.bool(True),
    estimator = cms.string('Chi2'),
    intermediateCleaning = cms.bool(True),
    lostHitPenalty = cms.double(30.0),
    maxCand = cms.int32(5),
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    seedAs5DHit = cms.bool(False),
    trajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('CkfBaseTrajectoryFilter_block')
    ),
    updator = cms.string('KFUpdator')
)
