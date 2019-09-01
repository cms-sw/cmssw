import FWCore.ParameterSet.Config as cms

PropagatorWithMaterialForMTD = cms.ESProducer(
    "PropagatorWithMaterialESProducer",
    MaxDPhi = cms.double(1.6),     #default was 1.6
    ComponentName = cms.string('PropagatorWithMaterialForMTD'),
    Mass = cms.double(0.13957018),     #default was 0.105
    PropagationDirection = cms.string('anyDirection'),
    useRungeKutta = cms.bool(False),           
    ptMin = cms.double(0.1),
    useOldAnalPropLogic = cms.bool(False)
)

