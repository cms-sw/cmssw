import FWCore.ParameterSet.Config as cms
from RecoParticleFlow.PFClusterProducer.particleFlowClusterECAL_cff import *

particleFlowClusterOOTECAL = particleFlowClusterECAL.clone()
particleFlowClusterOOTECAL.energyCorrector.applyMVACorrections = True
particleFlowClusterOOTECAL.energyCorrector.maxPtForMVAEvaluation = 90.
particleFlowClusterOOTECAL.inputECAL = cms.InputTag("particleFlowClusterOOTECALUncorrected")
