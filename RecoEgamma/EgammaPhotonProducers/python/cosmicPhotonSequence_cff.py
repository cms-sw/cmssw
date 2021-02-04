import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaPhotonProducers.photonSequence_cff import *
photonCore.scHybridBarrelProducer = "cosmicSuperClusters:CosmicBarrelSuperClusters"
photonCore.scIslandEndcapProducer = "cosmicSuperClusters:CosmicEndcapSuperClusters"
photonCore.minSCEt = 0.0


photons.usePrimaryVertex = False
photons.minSCEtBarrel    = 0.0
photons.minSCEtEndcap    = 0.0

cosmicPhotonTask = cms.Task(photonCore,photons)
cosmicPhotonSequence = cms.Sequence(cosmicPhotonTask)
