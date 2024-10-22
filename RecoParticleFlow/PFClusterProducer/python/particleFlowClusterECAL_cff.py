import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFClusterProducer.particleFlowClusterECAL_cfi import particleFlowClusterECAL
particleFlowClusterECAL.energyCorrector.applyMVACorrections = True
particleFlowClusterECAL.energyCorrector.maxPtForMVAEvaluation = 90.


from Configuration.Eras.Modifier_run2_ECAL_2016_cff import run2_ECAL_2016
from Configuration.Eras.Modifier_run2_ECAL_2017_cff import run2_ECAL_2017

(run2_ECAL_2016 | run2_ECAL_2017).toModify(particleFlowClusterECAL,
                        energyCorrector = dict(srfAwareCorrection = True, maxPtForMVAEvaluation = 300.))

