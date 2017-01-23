import FWCore.ParameterSet.Config as cms


from RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff import particleFlowRecHitECAL
from RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff import particleFlowClusterECAL
from RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff import particleFlowRecHitPS

#note I have not been careful in the slightest about the PS clustering as I dont really care about it
#so it could be completely messed up for all I know
#I only care about the barrel, endcap could be messed up so be warned

particleFlowRecHitECALGSFixed=particleFlowRecHitECAL.clone()
particleFlowRecHitPSGSFixed=particleFlowRecHitPS.clone()

particleFlowRecHitECALGSFixed.producers[0].src=cms.InputTag("ecalMultiAndGSWeightsRecHitsEB")
particleFlowRecHitECALGSFixed.producers[1].src=cms.InputTag("reducedEcalRecHitsEE")
particleFlowRecHitPSGSFixed.producers[0].src=cms.InputTag("reducedEcalRecHitsES")
particleFlowClusterECALGSFixed.energyCorrector.recHitsEBLabel=cms.InputTag("ecalMultiAndGSWeightsRecHitsEB")
particleFlowClusterECALGSFixed.energyCorrector.recHitsEELabel=cms.InputTag("reducedEcalRecHitsEE")


from RecoEcal.EgammaClusterProducers.particleFlowSuperClusteringSequence_cff import particleFlowSuperClusterECAL
particleFlowSuperClusterECALGSFixed=particleFlowSuperClusterECAL.clone()
particleFlowSuperClusterECALGSFixed.regressionConfig.ecalRecHitsEB=cms.InputTag("ecalMultiAndGSWeightsRecHitEB")
particleFlowSuperClusterECALGSFixed.regressionConfig.ecalRecHitsEE=cms.InputTag("reducedEcalRecHitsEE")

gsFixedParticleFlowSuperClustering = cms.Sequence(particleFlowRecHitECALGSFixed*
                                                  particleFlowRecHitPSGSFixed*
                                                  particleFlowClusterECALGSFixed*
                                                  particleFlowSuperClusterECALGSFixed)
