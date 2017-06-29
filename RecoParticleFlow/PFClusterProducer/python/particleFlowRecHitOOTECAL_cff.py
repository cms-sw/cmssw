import FWCore.ParameterSet.Config as cms
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitECAL_cfi import *

particleFlowRecHitOOTECAL = particleFlowRecHitECAL.clone()
particleFlowRecHitOOTECAL.producers[0].qualityTests[1].timingCleaning = False
particleFlowRecHitOOTECAL.producers[1].qualityTests[1].timingCleaning = False

from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy

## EB
run2_miniAOD_80XLegacy.toModify(
    particleFlowRecHitOOTECAL.producers[0], 
    src = "reducedEcalRecHitsEB",
    srFlags = ""
)
run2_miniAOD_80XLegacy.toModify(
    particleFlowRecHitOOTECAL.producers[0].qualityTests[0], 
    name = "PFRecHitQTestThreshold",
    threshold = cms.double(0.08),
    thresholds = None
) # from CMSSW_8_0_24: RecoParticleFlow/PFClusterProducer/python/particleFlowRecHitECAL_cfi.py

## EE
run2_miniAOD_80XLegacy.toModify(
    particleFlowRecHitOOTECAL.producers[1],
    src = "reducedEcalRecHitsEE",
    srFlags = ""
)
run2_miniAOD_80XLegacy.toModify(
    particleFlowRecHitOOTECAL.producers[1].qualityTests[0], 
    name = "PFRecHitQTestThreshold",
    threshold = cms.double(0.3),
    thresholds = None
) # from CMSSW_8_0_24: RecoParticleFlow/PFClusterProducer/python/particleFlowRecHitECAL_cfi.py
