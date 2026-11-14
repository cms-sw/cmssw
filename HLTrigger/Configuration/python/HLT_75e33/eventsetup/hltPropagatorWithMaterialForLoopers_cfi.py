import FWCore.ParameterSet.Config as cms

hltPropagatorWithMaterialForLoopers = cms.ESProducer("PropagatorWithMaterialESProducer",
    ComponentName = cms.string('hltPropagatorWithMaterialForLoopers'),
    Mass = cms.double(0.1396),
    MaxDPhi = cms.double(4.0),
    PropagationDirection = cms.string('alongMomentum'),
    SimpleMagneticField = cms.string(''),
    ptMin = cms.double(-1),
    useOldAnalPropLogic = cms.bool(False),
    useRungeKutta = cms.bool(False)
)
