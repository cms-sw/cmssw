import FWCore.ParameterSet.Config as cms

KFTrajectorySmootherForInOut = cms.ESProducer("KFTrajectorySmootherESProducer",
    ComponentName = cms.string('KFSmootherForInOut'),
    Estimator = cms.string('Chi2ForInOut'),
    Propagator = cms.string('oppositeToMomElePropagator'),
    RecoGeometry = cms.string('GlobalDetLayerGeometry'),
    Updator = cms.string('KFUpdator'),
    appendToDataLabel = cms.string(''),
    errorRescaling = cms.double(100),
    minHits = cms.int32(3)
)
