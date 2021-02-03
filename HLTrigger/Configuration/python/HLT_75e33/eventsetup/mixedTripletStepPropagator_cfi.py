import FWCore.ParameterSet.Config as cms

mixedTripletStepPropagator = cms.ESProducer("PropagatorWithMaterialESProducer",
    ComponentName = cms.string('mixedTripletStepPropagator'),
    Mass = cms.double(0.105),
    MaxDPhi = cms.double(1.6),
    PropagationDirection = cms.string('alongMomentum'),
    SimpleMagneticField = cms.string(''),
    ptMin = cms.double(0.1),
    useRungeKutta = cms.bool(False)
)
