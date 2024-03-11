import FWCore.ParameterSet.Config as cms

hgcalMergeLayerClusters = cms.EDProducer("MergeClusterProducer",
    layerClustersEE = cms.InputTag("hgcalLayerClustersEE"),
    layerClustersHSci = cms.InputTag("hgcalLayerClustersHSci"),
    layerClustersHSi = cms.InputTag("hgcalLayerClustersHSi"),
    mightGet = cms.optional.untracked.vstring,
    timeClname = cms.string('timeLayerCluster'),
    time_layerclustersEE = cms.InputTag("hgcalLayerClustersEE","timeLayerCluster"),
    time_layerclustersHSci = cms.InputTag("hgcalLayerClustersHSci","timeLayerCluster"),
    time_layerclustersHSi = cms.InputTag("hgcalLayerClustersHSi","timeLayerCluster")
)
