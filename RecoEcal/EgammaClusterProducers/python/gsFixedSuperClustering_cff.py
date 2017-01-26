import FWCore.ParameterSet.Config as cms


from RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff import particleFlowRecHitECAL
from RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff import particleFlowClusterECAL
from RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff import particleFlowClusterECALUncorrected
from RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff import particleFlowClusterPS
from RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff import particleFlowRecHitPS

#note I have not been careful in the slightest about the PS clustering as I dont really care about it
#so it could be completely messed up for all I know
#likely dont even need to remake them, will prob ditch in later versions
#I only care about the barrel, endcap could be messed up so be warned
particleFlowRecHitECALGSFixed=particleFlowRecHitECAL.clone()
particleFlowRecHitPSGSFixed=particleFlowRecHitPS.clone()

particleFlowRecHitECALGSFixed.producers[0].src=cms.InputTag("ecalMultiAndGSWeightRecHitEB")
particleFlowRecHitECALGSFixed.producers[1].src=cms.InputTag("reducedEcalRecHitsEE")
particleFlowRecHitPSGSFixed.producers[0].src=cms.InputTag("reducedEcalRecHitsES")

#meh, naming them gainswitched when really its just a re-run on the reduced ES rec hits
from RecoParticleFlow.PFClusterProducer.particleFlowClusterPS_cfi import particleFlowClusterPS
particleFlowClusterPSGSFixed = particleFlowClusterPS.clone()
particleFlowClusterPSGSFixed.recHitsSource = cms.InputTag("particleFlowRecHitPSGSFixed")

particleFlowClusterECALUncorrectedGSFixed = particleFlowClusterECALUncorrected.clone()
particleFlowClusterECALUncorrectedGSFixed.recHitsSource = cms.InputTag("particleFlowRecHitECALGSFixed")

particleFlowClusterECALGSFixed = particleFlowClusterECAL.clone()
particleFlowClusterECALGSFixed.energyCorrector.recHitsEBLabel=cms.InputTag("ecalMultiAndGSWeightRecHitEB")
particleFlowClusterECALGSFixed.energyCorrector.recHitsEELabel=cms.InputTag("reducedEcalRecHitsEE")
particleFlowClusterECALGSFixed.inputECAL = cms.InputTag("particleFlowClusterECALUncorrectedGSFixed")
particleFlowClusterECALGSFixed.inputPS = cms.InputTag("particleFlowClusterPSGSFixed")

from RecoEcal.EgammaClusterProducers.particleFlowSuperClusteringSequence_cff import particleFlowSuperClusterECAL
particleFlowSuperClusterECALGSFixed=particleFlowSuperClusterECAL.clone()
particleFlowSuperClusterECALGSFixed.regressionConfig.ecalRecHitsEB=cms.InputTag("ecalMultiAndGSWeightRecHitEB")
particleFlowSuperClusterECALGSFixed.regressionConfig.ecalRecHitsEE=cms.InputTag("reducedEcalRecHitsEE")
particleFlowSuperClusterECALGSFixed.PFClusters = cms.InputTag("particleFlowClusterECALGSFixed")
particleFlowSuperClusterECALGSFixed.ESAssociation = cms.InputTag("particleFlowClusterECALGSFixed")

gsFixedParticleFlowSuperClustering = cms.Sequence(particleFlowRecHitECALGSFixed*
                                                  particleFlowRecHitPSGSFixed*
                                                  particleFlowClusterPSGSFixed*
                                                  particleFlowClusterECALUncorrectedGSFixed*
                                                  particleFlowClusterECALGSFixed*
                                                  particleFlowSuperClusterECALGSFixed)
