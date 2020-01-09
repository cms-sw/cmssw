import FWCore.ParameterSet.Config as cms
from RecoParticleFlow.PFClusterProducer.particleFlowClusterECAL_cff import *

particleFlowClusterOOTECAL = particleFlowClusterECAL.clone()
particleFlowClusterOOTECAL.inputECAL = cms.InputTag("particleFlowClusterOOTECALUncorrected")

from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
run2_miniAOD_80XLegacy.toModify(particleFlowClusterOOTECAL,
                                energyCorrector = dict( 
                                    maxPtForMVAEvaluation = 90., srfAwareCorrection = False, 
                                    recHitsEBLabel = "reducedEcalRecHitsEB", recHitsEELabel = "reducedEcalRecHitsEE"
                                )
                            )
