import FWCore.ParameterSet.Config as cms

ticlMultiClustersFromTrackstersEM = cms.EDProducer("MultiClustersFromTrackstersProducer",
    LayerClusters = cms.InputTag("hgcalLayerClusters"),
    Tracksters = cms.InputTag("ticlTrackstersEM"),
    mightGet = cms.optional.untracked.vstring,
    verbosity = cms.untracked.uint32(3)
)
