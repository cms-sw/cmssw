import FWCore.ParameterSet.Config as cms

ticlMultiClustersFromTrackstersHADForEgamma = cms.EDProducer("MultiClustersFromTrackstersProducer",
    LayerClusters = cms.InputTag("hgcalLayerClusters"),
    Tracksters = cms.InputTag("ticlTrackstersHADForEgamma"),
    mightGet = cms.optional.untracked.vstring,
    verbosity = cms.untracked.uint32(3)
)
