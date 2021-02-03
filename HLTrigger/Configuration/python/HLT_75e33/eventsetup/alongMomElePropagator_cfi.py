import FWCore.ParameterSet.Config as cms

alongMomElePropagator = cms.ESProducer("PropagatorWithMaterialESProducer",
    ComponentName = cms.string('alongMomElePropagator'),
    Mass = cms.double(0.000511),
    MaxDPhi = cms.double(1.6),
    PropagationDirection = cms.string('alongMomentum'),
    SimpleMagneticField = cms.string(''),
    ptMin = cms.double(-1.0),
    useRungeKutta = cms.bool(False)
)
