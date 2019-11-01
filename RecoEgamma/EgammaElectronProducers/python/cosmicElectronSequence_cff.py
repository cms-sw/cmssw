import FWCore.ParameterSet.Config as cms

# includes for tracking rechits
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.StripCPE_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
from RecoTracker.TkNavigation.NavigationSchoolESProducer_cff import *
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi import *
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *

# module to produce electron seeds
from RecoEgamma.EgammaElectronProducers.ecalDrivenElectronSeeds_cff import *
from RecoEgamma.EgammaElectronProducers.gsfElectronSequence_cff import *


ecalDrivenElectronSeeds.barrelSuperClusters = cms.InputTag('cosmicSuperClusters','CosmicBarrelSuperClusters')
ecalDrivenElectronSeeds.endcapSuperClusters = cms.InputTag('cosmicSuperClusters','CosmicEndcapSuperClusters')
ecalDrivenElectronSeeds.SeedConfiguration.fromTrackerSeeds = cms.bool(False)

gsfElectrons.barrelSuperClusters = cms.InputTag('cosmicSuperClusters','CosmicBarrelSuperClusters')
gsfElectrons.endcapSuperClusters = cms.InputTag('cosmicSuperClusters','CosmicEndcapSuperClusters')
gsfElectrons.barrelClusterShapes = cms.InputTag('cosmicSuperClusters','CosmicBarrelSuperClusters')
gsfElectrons.endcapClusterShapes = cms.InputTag('cosmicSuperClusters','CosmicEndcapSuperClusters')

cosmicElectronSequence = cms.Sequence(gsfElectronSequence)
