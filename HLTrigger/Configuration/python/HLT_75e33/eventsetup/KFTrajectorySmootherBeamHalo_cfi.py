import FWCore.ParameterSet.Config as cms

KFTrajectorySmootherBeamHalo = cms.ESProducer("KFTrajectorySmootherESProducer",
    ComponentName = cms.string('KFSmootherBH'),
    Estimator = cms.string('Chi2'),
    Propagator = cms.string('BeamHaloPropagatorAlong'),
    RecoGeometry = cms.string('GlobalDetLayerGeometry'),
    Updator = cms.string('KFUpdator'),
    appendToDataLabel = cms.string(''),
    errorRescaling = cms.double(100),
    minHits = cms.int32(3)
)
