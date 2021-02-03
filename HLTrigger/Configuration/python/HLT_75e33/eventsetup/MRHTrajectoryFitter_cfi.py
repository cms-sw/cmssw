import FWCore.ParameterSet.Config as cms

MRHTrajectoryFitter = cms.ESProducer("KFTrajectoryFitterESProducer",
    ComponentName = cms.string('MRHFitter'),
    Estimator = cms.string('MRHChi2'),
    Propagator = cms.string('RungeKuttaTrackerPropagator'),
    RecoGeometry = cms.string('GlobalDetLayerGeometry'),
    Updator = cms.string('KFUpdator'),
    appendToDataLabel = cms.string(''),
    minHits = cms.int32(3)
)
