import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaElectronProducers.electronSequence_cff import *
electronPixelSeeds.barrelSuperClusters = cms.InputTag('cosmicSuperClusters','CosmicBarrelSuperClusters')
electronPixelSeeds.endcapSuperClusters = cms.InputTag('cosmicSuperClusters','CosmicEndcapSuperClusters')
electronPixelSeeds.SeedConfiguration.fromTrackerSeeds = cms.bool(False)

pixelMatchGsfElectrons.barrelSuperClusters = cms.InputTag('cosmicSuperClusters','CosmicBarrelSuperClusters')
pixelMatchGsfElectrons.endcapSuperClusters = cms.InputTag('cosmicSuperClusters','CosmicEndcapSuperClusters')
pixelMatchGsfElectrons.barrelClusterShapes = cms.InputTag('cosmicSuperClusters','CosmicBarrelSuperClusters')
pixelMatchGsfElectrons.endcapClusterShapes = cms.InputTag('cosmicSuperClusters','CosmicEndcapSuperClusters')

cosmicElectronSequence = cms.Sequence(electronSequence)
