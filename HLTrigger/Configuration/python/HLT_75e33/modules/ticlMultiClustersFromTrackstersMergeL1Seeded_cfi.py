import FWCore.ParameterSet.Config as cms

ticlMultiClustersFromTrackstersMergeL1Seeded = cms.EDProducer("MultiClustersFromTrackstersProducer",
    LayerClusters = cms.InputTag("hgcalLayerClustersL1Seeded"),
    Tracksters = cms.InputTag("ticlTrackstersEML1"),
    mightGet = cms.optional.untracked.vstring,
    verbosity = cms.untracked.uint32(3)
)
