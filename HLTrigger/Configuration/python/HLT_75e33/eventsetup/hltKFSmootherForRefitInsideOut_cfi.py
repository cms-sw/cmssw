import FWCore.ParameterSet.Config as cms

hltKFSmootherForRefitInsideOut = cms.ESProducer('KFTrajectorySmootherESProducer',
    ComponentName = cms.string('hltKFSmootherForRefitInsideOut'),
    Propagator = cms.string('hltSmartPropagatorAnyRK'),
    Updator = cms.string('hltESPKFUpdator'),
    Estimator = cms.string('hltChi2EstimatorForRefit'),
    RecoGeometry = cms.string('GlobalDetLayerGeometry'),
    errorRescaling = cms.double(100),
    minHits = cms.int32(3),
    appendToDataLabel = cms.string('')
)
