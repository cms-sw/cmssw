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

## EE
run2_miniAOD_80XLegacy.toModify(
    particleFlowRecHitOOTECAL.producers[1],
    src = "reducedEcalRecHitsEE",
    srFlags = ""
)
