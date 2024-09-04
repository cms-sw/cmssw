import FWCore.ParameterSet.Config as cms

hltTrackstersSoAProducer = cms.EDProducer('TrackstersSoAProducer@alpaka',
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    ),
    layer_clusters = cms.InputTag("hltHgcalSoALayerClustersProducer"),
    filtered_clusters_mask = cms.InputTag("hltFilteredLayerClustersSoAProducer"),
    layer_clusters_tiles = cms.InputTag("ticlLayerTileProducer"),
    seeding_regions = cms.InputTag("ticlSeedingGlobal")
)
