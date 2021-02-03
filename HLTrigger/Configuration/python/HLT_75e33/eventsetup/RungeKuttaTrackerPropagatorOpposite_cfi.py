import FWCore.ParameterSet.Config as cms

RungeKuttaTrackerPropagatorOpposite = cms.ESProducer("PropagatorWithMaterialESProducer",
    ComponentName = cms.string('RungeKuttaTrackerPropagatorOpposite'),
    Mass = cms.double(0.105),
    MaxDPhi = cms.double(1.6),
    PropagationDirection = cms.string('oppositeToMomentum'),
    SimpleMagneticField = cms.string(''),
    ptMin = cms.double(-1.0),
    useRungeKutta = cms.bool(True)
)
