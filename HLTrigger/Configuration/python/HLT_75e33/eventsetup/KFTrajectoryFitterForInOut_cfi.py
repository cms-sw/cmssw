import FWCore.ParameterSet.Config as cms

KFTrajectoryFitterForInOut = cms.ESProducer("KFTrajectoryFitterESProducer",
    ComponentName = cms.string('KFFitterForInOut'),
    Estimator = cms.string('Chi2ForInOut'),
    Propagator = cms.string('alongMomElePropagator'),
    RecoGeometry = cms.string('GlobalDetLayerGeometry'),
    Updator = cms.string('KFUpdator'),
    appendToDataLabel = cms.string(''),
    minHits = cms.int32(3)
)
