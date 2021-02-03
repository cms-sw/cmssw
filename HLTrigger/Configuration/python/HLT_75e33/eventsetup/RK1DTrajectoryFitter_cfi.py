import FWCore.ParameterSet.Config as cms

RK1DTrajectoryFitter = cms.ESProducer("KFTrajectoryFitterESProducer",
    ComponentName = cms.string('RK1DFitter'),
    Estimator = cms.string('Chi2'),
    Propagator = cms.string('RungeKuttaTrackerPropagator'),
    RecoGeometry = cms.string('GlobalDetLayerGeometry'),
    Updator = cms.string('KFSwitching1DUpdator'),
    appendToDataLabel = cms.string(''),
    minHits = cms.int32(3)
)
