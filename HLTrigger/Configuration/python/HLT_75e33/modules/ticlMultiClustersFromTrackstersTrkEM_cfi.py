import FWCore.ParameterSet.Config as cms

ticlMultiClustersFromTrackstersTrkEM = cms.EDProducer("MultiClustersFromTrackstersProducer",
    LayerClusters = cms.InputTag("hgcalLayerClusters"),
    Tracksters = cms.InputTag("ticlTrackstersTrkEM"),
    mightGet = cms.optional.untracked.vstring,
    verbosity = cms.untracked.uint32(3)
)
