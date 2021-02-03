import FWCore.ParameterSet.Config as cms

CkfTrajectoryBuilderBeamHalo = cms.PSet(
    ComponentType = cms.string('CkfTrajectoryBuilder'),
    MeasurementTrackerName = cms.string(''),
    TTRHBuilder = cms.string('WithTrackAngle'),
    alwaysUseInvalidHits = cms.bool(True),
    estimator = cms.string('Chi2'),
    intermediateCleaning = cms.bool(True),
    lostHitPenalty = cms.double(30.0),
    maxCand = cms.int32(5),
    propagatorAlong = cms.string('BeamHaloPropagatorAlong'),
    propagatorOpposite = cms.string('BeamHaloPropagatorOpposite'),
    seedAs5DHit = cms.bool(False),
    trajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('ckfTrajectoryFilterBeamHaloMuon')
    ),
    updator = cms.string('KFUpdator')
)