import FWCore.ParameterSet.Config as cms

hltHgcalMergeLayerClustersL1Seeded = cms.EDProducer("MergeClusterProducer",
    layerClustersEE = cms.InputTag("hltHgcalLayerClustersEEL1Seeded"),
    layerClustersHSci = cms.InputTag("hltHgcalLayerClustersHSciL1Seeded"),
    layerClustersHSi = cms.InputTag("hltHgcalLayerClustersHSiL1Seeded"),
    mightGet = cms.optional.untracked.vstring,
    timeClname = cms.string('timeLayerCluster'),
    time_layerclustersEE = cms.InputTag("hltHgcalLayerClustersEEL1Seeded","timeLayerCluster"),
    time_layerclustersHSci = cms.InputTag("hltHgcalLayerClustersHSciL1Seeded","timeLayerCluster"),
    time_layerclustersHSi = cms.InputTag("hltHgcalLayerClustersHSiL1Seeded","timeLayerCluster")
)
