import FWCore.ParameterSet.Config as cms
from RecoParticleFlow.PFClusterProducer.particleFlowClusterECAL_cff import *

particleFlowClusterOOTECAL = particleFlowClusterECAL.clone()
particleFlowClusterOOTECAL.energyCorrector.applyMVACorrections = cms.bool(True)
particleFlowClusterOOTECAL.energyCorrector.maxPtForMVAEvaluation = cms.double(90.)
particleFlowClusterOOTECAL.inputECAL = cms.InputTag("particleFlowClusterOOTECALUncorrected")
