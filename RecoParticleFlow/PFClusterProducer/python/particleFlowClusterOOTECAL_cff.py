import FWCore.ParameterSet.Config as cms
from RecoParticleFlow.PFClusterProducer.particleFlowClusterECAL_cfi import *

particleFlowClusterOOTECAL = particleFlowClusterECAL.clone()
particleFlowClusterOOTECAL.inputECAL = cms.InputTag("particleFlowClusterOOTECALUncorrected")

from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
run2_miniAOD_80XLegacy.toModify(
    particleFlowClusterOOTECAL.energyCorrector, 
    recHitsEBLabel = "reducedEcalRecHitsEB",
    recHitsEELabel = "reducedEcalRecHitsEE"
)
