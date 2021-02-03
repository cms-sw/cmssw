import FWCore.ParameterSet.Config as cms

TrajectoryBuilderForElectrons = cms.PSet(
    ComponentType = cms.string('CkfTrajectoryBuilder'),
    MeasurementTrackerName = cms.string(''),
    TTRHBuilder = cms.string('WithTrackAngle'),
    alwaysUseInvalidHits = cms.bool(True),
    estimator = cms.string('ElectronChi2'),
    intermediateCleaning = cms.bool(False),
    lostHitPenalty = cms.double(90.0),
    maxCand = cms.int32(5),
    propagatorAlong = cms.string('fwdGsfElectronPropagator'),
    propagatorOpposite = cms.string('bwdGsfElectronPropagator'),
    seedAs5DHit = cms.bool(False),
    trajectoryFilter = cms.PSet(
        refToPSet_ = cms.string('TrajectoryFilterForElectrons')
    ),
    updator = cms.string('KFUpdator')
)