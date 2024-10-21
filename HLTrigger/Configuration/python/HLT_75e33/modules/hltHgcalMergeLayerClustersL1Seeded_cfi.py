import FWCore.ParameterSet.Config as cms

hltHgcalMergeLayerClustersL1Seeded = cms.EDProducer("MergeClusterProducer",
    layerClustersEE = cms.InputTag("hltHgcalLayerClustersEEL1Seeded"),
    layerClustersHSci = cms.InputTag("hltHgcalLayerClustersHSciL1Seeded"),
    layerClustersHSi = cms.InputTag("hltHgcalLayerClustersHSiL1Seeded"),
    layerClustersEB = cms.InputTag("hltBarrelLayerClustersEBL1Seeded"),
    layerClustersHB = cms.InputTag("hltBarrelLayerClustersEBL1Seeded"),
    mightGet = cms.optional.untracked.vstring,
    timeClname = cms.string('timeLayerCluster'),
    time_layerclustersEE = cms.InputTag("hltHgcalLayerClustersEEL1Seeded","timeLayerCluster"),
    time_layerclustersHSci = cms.InputTag("hltHgcalLayerClustersHSciL1Seeded","timeLayerCluster"),
    time_layerclustersHSi = cms.InputTag("hltHgcalLayerClustersHSiL1Seeded","timeLayerCluster"),
    time_layerclustersEB = cms.InputTag("hltBarrelLayerClustersEBL1Seeded","timeLayerCluster"),
    time_layerclustersHB = cms.InputTag("hltBarrelLayerClustersEBL1Seeded","timeLayerCluster"),
)
