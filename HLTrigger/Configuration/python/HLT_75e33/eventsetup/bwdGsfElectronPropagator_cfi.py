import FWCore.ParameterSet.Config as cms

bwdGsfElectronPropagator = cms.ESProducer("PropagatorWithMaterialESProducer",
    ComponentName = cms.string('bwdGsfElectronPropagator'),
    Mass = cms.double(0.000511),
    MaxDPhi = cms.double(1.6),
    PropagationDirection = cms.string('oppositeToMomentum'),
    SimpleMagneticField = cms.string(''),
    ptMin = cms.double(-1.0),
    useRungeKutta = cms.bool(False)
)
