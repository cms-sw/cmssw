import FWCore.ParameterSet.Config as cms

CkfTrajectoryBuilder = cms.PSet(
    ComponentType = cms.string('CkfTrajectoryBuilder'),
    propagatorAlong = cms.string('PropagatorWithMaterial'),
#    propagatorAlong = cms.string('PropagatorWithMaterialParabolicMf'),
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('CkfBaseTrajectoryFilter_block')),
    maxCand = cms.int32(5),
    intermediateCleaning = cms.bool(True),
    MeasurementTrackerName = cms.string(''),
    estimator = cms.string('Chi2'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    updator = cms.string('KFUpdator'),
    alwaysUseInvalidHits = cms.bool(True),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
#    propagatorOpposite = cms.string('PropagatorWithMaterialParabolicMfOpposite'),
    lostHitPenalty = cms.double(30.0),
    #SharedSeedCheck = cms.bool(False)
)


