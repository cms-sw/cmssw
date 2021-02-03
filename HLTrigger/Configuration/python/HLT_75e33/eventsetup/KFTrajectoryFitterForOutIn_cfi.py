import FWCore.ParameterSet.Config as cms

KFTrajectoryFitterForOutIn = cms.ESProducer("KFTrajectoryFitterESProducer",
    ComponentName = cms.string('KFFitterForOutIn'),
    Estimator = cms.string('Chi2ForOutIn'),
    Propagator = cms.string('alongMomElePropagator'),
    RecoGeometry = cms.string('GlobalDetLayerGeometry'),
    Updator = cms.string('KFUpdator'),
    appendToDataLabel = cms.string(''),
    minHits = cms.int32(3)
)
