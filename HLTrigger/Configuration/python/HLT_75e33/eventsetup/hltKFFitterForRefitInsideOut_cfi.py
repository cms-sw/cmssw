import FWCore.ParameterSet.Config as cms

hltKFFitterForRefitInsideOut = cms.ESProducer('KFTrajectoryFitterESProducer',
    RecoGeometry = cms.string('GlobalDetLayerGeometry'),
    ComponentName = cms.string('hltKFFitterForRefitInsideOut'),
    Propagator = cms.string('hltSmartPropagatorAnyRK'),
    Updator = cms.string('hltESPKFUpdator'),
    Estimator = cms.string('hltChi2EstimatorForRefit'),
    minHits = cms.int32(3),
    appendToDataLabel = cms.string('')
)
