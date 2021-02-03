import FWCore.ParameterSet.Config as cms

ticlMultiClustersFromTrackstersMerge = cms.EDProducer("MultiClustersFromTrackstersProducer",
    LayerClusters = cms.InputTag("hgcalLayerClusters"),
    Tracksters = cms.InputTag("ticlTrackstersMerge"),
    mightGet = cms.optional.untracked.vstring,
    verbosity = cms.untracked.uint32(3)
)
