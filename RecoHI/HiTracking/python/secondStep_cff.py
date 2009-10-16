import FWCore.ParameterSet.Config as cms

#################################
# Remaining clusters
secondClusters = cms.EDFilter("TrackClusterRemover",
    trajectories = cms.InputTag("hiSelectedTracks"),
    pixelClusters = cms.InputTag("siPixelClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(999999.0)
    ),
    stripClusters = cms.InputTag("siStripClusters")
)

# Remaining pixel hits
import RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi
secondPixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone()
secondPixelRecHits.src = 'secondClusters:'

# Remaining strip hits
import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi
secondStripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone()
secondStripRecHits.ClusterProducer = 'secondClusters'

#################################
# Pixel Pair Layers
import RecoTracker.TkSeedingLayers.PixelLayerPairs_cfi
newPixelLayerPairs = RecoTracker.TkSeedingLayers.PixelLayerPairs_cfi.pixellayerpairs.clone(
    ComponentName = 'newPixelLayerPairs',
    )
newPixelLayerPairs.BPix.HitProducer = 'secondPixelRecHits'
newPixelLayerPairs.FPix.HitProducer = 'secondPixelRecHits'

# Pixel Pair Seeding
from RecoTracker.TkSeedGenerator.GlobalSeedsFromPairsWithVertices_cff import *
newSeedFromPairs = RecoTracker.TkSeedGenerator.GlobalSeedsFromPairsWithVertices_cff.globalSeedsFromPairsWithVertices.clone()
newSeedFromPairs.RegionFactoryPSet.RegionPSet.ptMin = 2.0
newSeedFromPairs.RegionFactoryPSet.RegionPSet.originRadius = 0.05
newSeedFromPairs.RegionFactoryPSet.RegionPSet.fixedError=0.2
newSeedFromPairs.RegionFactoryPSet.RegionPSet.VertexCollection=cms.InputTag("hiSelectedVertex")
newSeedFromPairs.OrderedHitsFactoryPSet.SeedingLayers = cms.string('newPixelLayerPairs')
newSeedFromPairs.ClusterCheckPSet.doClusterCheck=False


secondStep = cms.Sequence(secondClusters * secondPixelRecHits * secondStripRecHits * newSeedFromPairs)
