import FWCore.ParameterSet.Config as cms

hgcalMergeLayerClustersL1Seeded = cms.EDProducer("MergeClusterProducer",
    layerClustersEE = cms.InputTag("hgcalLayerClustersEEL1Seeded"),
    layerClustersHSci = cms.InputTag("hgcalLayerClustersHSciL1Seeded"),
    layerClustersHSi = cms.InputTag("hgcalLayerClustersHSiL1Seeded"),
    mightGet = cms.optional.untracked.vstring,
    timeClname = cms.string('timeLayerCluster'),
    time_layerclustersEE = cms.InputTag("hgcalLayerClustersEEL1Seeded","timeLayerCluster"),
    time_layerclustersHSci = cms.InputTag("hgcalLayerClustersHSciL1Seeded","timeLayerCluster"),
    time_layerclustersHSi = cms.InputTag("hgcalLayerClustersHSiL1Seeded","timeLayerCluster")
)
