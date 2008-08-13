import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaPhotonProducers.photonSequence_cff import *
photons.scHybridBarrelProducer = cms.InputTag("cosmicSuperClusters","CosmicBarrelSuperClusters")
photons.scIslandEndcapProducer = cms.InputTag("cosmicSuperClusters","CosmicEndcapSuperClusters")


photons.usePrimaryVertex = cms.bool(False)
photons.minSCEt = cms.double(0.0)

cosmicPhotonSequence = cms.Sequence(photons)
