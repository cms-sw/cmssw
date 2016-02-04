import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaPhotonProducers.photonSequence_cff import *
photonCore.scHybridBarrelProducer = cms.InputTag("cosmicSuperClusters","CosmicBarrelSuperClusters")
photonCore.scIslandEndcapProducer = cms.InputTag("cosmicSuperClusters","CosmicEndcapSuperClusters")
photonCore.minSCEt = cms.double(0.0)


photons.usePrimaryVertex = cms.bool(False)
photons.minSCEtBarrel = cms.double(0.0)
photons.minSCEtEndcap = cms.double(0.0)

cosmicPhotonSequence = cms.Sequence(photonCore+photons)
