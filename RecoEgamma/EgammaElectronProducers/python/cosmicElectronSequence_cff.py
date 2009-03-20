import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaElectronProducers.electronSequence_cff import *
from RecoEgamma.EgammaElectronProducers.ecalDrivenElectronSeedsModules_cff import *

ecalDrivenElectronSeeds.barrelSuperClusters = cms.InputTag('cosmicSuperClusters','CosmicBarrelSuperClusters')
ecalDrivenElectronSeeds.endcapSuperClusters = cms.InputTag('cosmicSuperClusters','CosmicEndcapSuperClusters')
ecalDrivenElectronSeeds.SeedConfiguration.fromTrackerSeeds = cms.bool(False)

gsfElectrons.barrelSuperClusters = cms.InputTag('cosmicSuperClusters','CosmicBarrelSuperClusters')
gsfElectrons.endcapSuperClusters = cms.InputTag('cosmicSuperClusters','CosmicEndcapSuperClusters')
gsfElectrons.barrelClusterShapes = cms.InputTag('cosmicSuperClusters','CosmicBarrelSuperClusters')
gsfElectrons.endcapClusterShapes = cms.InputTag('cosmicSuperClusters','CosmicEndcapSuperClusters')

cosmicElectronSequence = cms.Sequence(electronSequence)
