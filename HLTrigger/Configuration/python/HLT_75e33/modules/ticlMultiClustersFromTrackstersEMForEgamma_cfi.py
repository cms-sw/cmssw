import FWCore.ParameterSet.Config as cms

ticlMultiClustersFromTrackstersEMForEgamma = cms.EDProducer("MultiClustersFromTrackstersProducer",
    LayerClusters = cms.InputTag("hgcalLayerClusters"),
    Tracksters = cms.InputTag("ticlTrackstersEMForEgamma"),
    mightGet = cms.optional.untracked.vstring,
    verbosity = cms.untracked.uint32(3)
)
