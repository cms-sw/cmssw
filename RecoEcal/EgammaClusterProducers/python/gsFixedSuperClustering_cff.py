import FWCore.ParameterSet.Config as cms


import RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff

#note I have not been careful in the slightest about the PS clustering as I dont really care about it
#so it could be completely messed up for all I know
#likely dont even need to remake them, will prob ditch in later versions
#I only care about the barrel, endcap could be messed up so be warned
particleFlowRecHitECALGSFixed=RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff.particleFlowRecHitECAL.clone()
particleFlowRecHitPSGSFixed=RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff.particleFlowRecHitPS.clone()

particleFlowRecHitECALGSFixed.producers[0].src=cms.InputTag("ecalMultiAndGSGlobalRecHitEB")
particleFlowRecHitECALGSFixed.producers[1].src=cms.InputTag("reducedEcalRecHitsEE")
particleFlowRecHitPSGSFixed.producers[0].src=cms.InputTag("reducedEcalRecHitsES")

#meh, naming them gainswitched when really its just a re-run on the reduced ES rec hits
particleFlowClusterPSGSFixed = RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff.particleFlowClusterPS.clone()
particleFlowClusterPSGSFixed.recHitsSource = cms.InputTag("particleFlowRecHitPSGSFixed")

particleFlowClusterECALUncorrectedGSFixed = RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff.particleFlowClusterECALUncorrected.clone()
particleFlowClusterECALUncorrectedGSFixed.recHitsSource = cms.InputTag("particleFlowRecHitECALGSFixed")

particleFlowClusterECALGSFixed = RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff.particleFlowClusterECAL.clone()
particleFlowClusterECALGSFixed.energyCorrector.recHitsEBLabel=cms.InputTag("ecalMultiAndGSGlobalRecHitEB")
particleFlowClusterECALGSFixed.energyCorrector.recHitsEELabel=cms.InputTag("reducedEcalRecHitsEE")
particleFlowClusterECALGSFixed.inputECAL = cms.InputTag("particleFlowClusterECALUncorrectedGSFixed")
particleFlowClusterECALGSFixed.inputPS = cms.InputTag("particleFlowClusterPSGSFixed")

import RecoEcal.EgammaClusterProducers.particleFlowSuperClusteringSequence_cff
particleFlowSuperClusterECALGSFixed=RecoEcal.EgammaClusterProducers.particleFlowSuperClusteringSequence_cff.particleFlowSuperClusterECAL.clone()
particleFlowSuperClusterECALGSFixed.regressionConfig.ecalRecHitsEB=cms.InputTag("ecalMultiAndGSGlobalRecHitEB")
particleFlowSuperClusterECALGSFixed.regressionConfig.ecalRecHitsEE=cms.InputTag("reducedEcalRecHitsEE")
particleFlowSuperClusterECALGSFixed.PFClusters = cms.InputTag("particleFlowClusterECALGSFixed")
particleFlowSuperClusterECALGSFixed.ESAssociation = cms.InputTag("particleFlowClusterECALGSFixed")

gsFixedParticleFlowSuperClustering = cms.Sequence(particleFlowRecHitECALGSFixed*
                                                  particleFlowRecHitPSGSFixed*
                                                  particleFlowClusterPSGSFixed*
                                                  particleFlowClusterECALUncorrectedGSFixed*
                                                  particleFlowClusterECALGSFixed*
                                                  particleFlowSuperClusterECALGSFixed)
