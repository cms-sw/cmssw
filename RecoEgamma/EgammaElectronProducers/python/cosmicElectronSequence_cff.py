import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaElectronProducers.electronSequence_cff import *
from RecoEgamma.EgammaElectronProducers.ecalDrivenElectronSeedsModules_cff import *

ecalDrivenElectronSeeds.barrelSuperClusters = cms.InputTag('cosmicSuperClusters','CosmicBarrelSuperClusters')
ecalDrivenElectronSeeds.endcapSuperClusters = cms.InputTag('cosmicSuperClusters','CosmicEndcapSuperClusters')
ecalDrivenElectronSeeds.SeedConfiguration.fromTrackerSeeds = cms.bool(False)

pixelMatchGsfElectrons.barrelSuperClusters = cms.InputTag('cosmicSuperClusters','CosmicBarrelSuperClusters')
pixelMatchGsfElectrons.endcapSuperClusters = cms.InputTag('cosmicSuperClusters','CosmicEndcapSuperClusters')
pixelMatchGsfElectrons.barrelClusterShapes = cms.InputTag('cosmicSuperClusters','CosmicBarrelSuperClusters')
pixelMatchGsfElectrons.endcapClusterShapes = cms.InputTag('cosmicSuperClusters','CosmicEndcapSuperClusters')

cosmicElectronSequence = cms.Sequence(electronSequence)
