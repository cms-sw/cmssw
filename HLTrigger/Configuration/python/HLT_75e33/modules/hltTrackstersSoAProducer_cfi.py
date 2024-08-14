import FWCore.ParameterSet.Config as cms

hltTrackstersSoAProducer = cms.EDProducer('TrackstersSoAProducer@alpaka',
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    ),
    layer_clusters = cms.InputTag("hltHgcalSoALayerClustersProducer"),
    clusters_mask_soa = cms.InputTag("hltFilteredLayerClustersSoAProducer")
)
