import FWCore.ParameterSet.Config as cms

hltHgcalMergeLayerClusters = cms.EDProducer("MergeClusterProducer",
    layerClusters = cms.VInputTag("hltHgcalLayerClustersEE", "hltHgcalLayerClustersHSci", "hltHgcalLayerClustersHSi"),
    mightGet = cms.optional.untracked.vstring,
    time_layerclusters = cms.VInputTag("hltHgcalLayerClustersEE:timeLayerCluster", "hltHgcalLayerClustersHSci:timeLayerCluster", "hltHgcalLayerClustersHSi:timeLayerCluster")
)

from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel
layerClusters = cms.VInputTag("hltHgcalLayerClustersEE", "hltHgcalLayerClustersHSci", "hltHgcalLayerClustersHSi", "hltBarrelLayerClustersEB", "hltBarrelLayerClustersHB")
time_layerclusters = cms.VInputTag("hltHgcalLayerClustersEE:timeLayerCluster", "hltHgcalLayerClustersHSci:timeLayerCluster", "hltHgcalLayerClustersHSi:timeLayerCluster", "hltBarrelLayerClustersEB:timeLayerCluster", "hltBarrelLayerClustersHB:timeLayerCluster")
ticl_barrel.toModify(hltHgcalMergeLayerClusters, layerClusters = layerClusters, time_layerclusters = time_layerclusters)
