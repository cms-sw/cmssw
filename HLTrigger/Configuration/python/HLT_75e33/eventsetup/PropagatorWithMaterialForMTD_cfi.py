import FWCore.ParameterSet.Config as cms

PropagatorWithMaterialForMTD = cms.ESProducer("PropagatorWithMaterialESProducer",
    ComponentName = cms.string('PropagatorWithMaterialForMTD'),
    Mass = cms.double(0.13957018),
    MaxDPhi = cms.double(1.6),
    PropagationDirection = cms.string('anyDirection'),
    ptMin = cms.double(0.1),
    useOldAnalPropLogic = cms.bool(False),
    useRungeKutta = cms.bool(False)
)
