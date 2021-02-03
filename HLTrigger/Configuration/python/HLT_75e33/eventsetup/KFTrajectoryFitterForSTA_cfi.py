import FWCore.ParameterSet.Config as cms

KFTrajectoryFitterForSTA = cms.ESProducer("KFTrajectoryFitterESProducer",
    ComponentName = cms.string('KFFitterSTA'),
    Estimator = cms.string('Chi2STA'),
    Propagator = cms.string('SteppingHelixPropagatorAny'),
    RecoGeometry = cms.string('GlobalDetLayerGeometry'),
    Updator = cms.string('KFUpdator'),
    appendToDataLabel = cms.string(''),
    minHits = cms.int32(3)
)
