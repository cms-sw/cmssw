import FWCore.ParameterSet.Config as cms

oppositeToMomElePropagator = cms.ESProducer("PropagatorWithMaterialESProducer",
    ComponentName = cms.string('oppositeToMomElePropagator'),
    Mass = cms.double(0.000511),
    MaxDPhi = cms.double(1.6),
    PropagationDirection = cms.string('oppositeToMomentum'),
    SimpleMagneticField = cms.string(''),
    ptMin = cms.double(-1.0),
    useRungeKutta = cms.bool(False)
)
