import FWCore.ParameterSet.Config as cms

def _addProcessPropagatorWithMaterialForMTD(process):
    process.hltPropagatorWithMaterialForMTD = cms.ESProducer(
        "PropagatorWithMaterialESProducer",
        MaxDPhi = cms.double(1.6),     #default was 1.6
        ComponentName = cms.string('hltPropagatorWithMaterialForMTD'),
        Mass = cms.double(0.13957018),     #default was 0.105
        PropagationDirection = cms.string('anyDirection'),
        useRungeKutta = cms.bool(False),
        ptMin = cms.double(0.1),
        useOldAnalPropLogic = cms.bool(False)
    )

from Configuration.ProcessModifiers.mtd_at_hlt_cff import mtd_at_hlt
modifyConfigurationForPropagatorWithMaterialForMTD_ = mtd_at_hlt.makeProcessModifier(_addProcessPropagatorWithMaterialForMTD)
