import FWCore.ParameterSet.Config as cms

hltHgcalMergeLayerClustersL1Seeded = cms.EDProducer("MergeClusterProducer",
    layerClusters = cms.VInputTag("hltHgcalLayerClustersEEL1Seeded", "hltHgcalLayerClustersHSciL1Seeded", "hltHgcalLayerClustersHSiL1Seeded"),
    mightGet = cms.optional.untracked.vstring,
    time_layerclusters = cms.VInputTag("hltHgcalLayerClustersEEL1Seeded:timeLayerCluster","hltHgcalLayerClustersHSciL1Seeded:timeLayerCluster","hltHgcalLayerClustersHSiL1Seeded:timeLayerCluster")
)

from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel
layerClusters = cms.VInputTag("hltHgcalLayerClustersEEL1Seeded", "hltHgcalLayerClustersHSciL1Seeded", "hltHgcalLayerClustersHSiL1Seeded", "hltBarrelLayerClustersEBL1Seeded")
time_layerclusters = cms.VInputTag("hltHgcalLayerClustersEEL1Seeded:timeLayerCluster", "hltHgcalLayerClustersHSciL1Seeded:timeLayerCluster", "hltHgcalLayerClustersHSiL1Seeded:timeLayerCluster", "hltBarrelLayerClustersEBL1Seeded:timeLayerCluster")
ticl_barrel.toModify(hltHgcalMergeLayerClustersL1Seeded, layerClusters = layerClusters, time_layerclusters = time_layerclusters)