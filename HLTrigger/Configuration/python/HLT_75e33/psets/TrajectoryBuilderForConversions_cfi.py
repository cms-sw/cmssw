import FWCore.ParameterSet.Config as cms

TrajectoryBuilderForConversions = cms.PSet(
    ComponentType = cms.string('CkfTrajectoryBuilder'),
    MeasurementTrackerName = cms.string(''),
    TTRHBuilder = cms.string('WithTrackAngle'),
    alwaysUseInvalidHits = cms.bool(True),
    estimator = cms.string('eleLooseChi2'),
    intermediateCleaning = cms.bool(True),
    lostHitPenalty = cms.double(30.0),
    maxCand = cms.int32(5),
    propagatorAlong = cms.string('alongMomElePropagator'),
    propagatorOpposite = cms.string('oppositeToMomElePropagator'),
    seedAs5DHit = cms.bool(False),
    trajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('TrajectoryFilterForConversions')
    ),
    updator = cms.string('KFUpdator')
)