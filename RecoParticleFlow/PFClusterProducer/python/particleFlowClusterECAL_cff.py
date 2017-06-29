import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFClusterProducer.particleFlowClusterECAL_cfi import particleFlowClusterECAL
particleFlowClusterECAL.energyCorrector.applyMVACorrections = cms.bool(True)
particleFlowClusterECAL.energyCorrector.maxPtForMVAEvaluation = cms.double(90.)

from Configuration.Eras.Modifier_run2_ECAL_2017_cff import run2_ECAL_2017
run2_ECAL_2017.toModify(particleFlowClusterECAL,
                        energyCorrector = dict(srfAwareCorrection = cms.bool(True), maxPtForMVAEvaluation = cms.double(300.)))

particleFlowClusterECALSequence = cms.Sequence(particleFlowClusterECAL)
