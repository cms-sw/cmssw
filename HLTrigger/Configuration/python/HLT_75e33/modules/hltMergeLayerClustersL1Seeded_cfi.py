import FWCore.ParameterSet.Config as cms

hltMergeLayerClustersL1Seeded = cms.EDProducer("MergeClusterProducer",
    layerClusters = cms.VInputTag("hltHgCalLayerClustersFromSoAProducerL1Seeded", "hltHgCalLayerClustersFromSoAProducerHSciL1Seeded", "hltHgCalLayerClustersFromSoAProducerHSiL1Seeded"),
    mightGet = cms.optional.untracked.vstring,
    time_layerclusters = cms.VInputTag("hltHgCalLayerClustersFromSoAProducerL1Seeded:timeLayerCluster","hltHgCalLayerClustersFromSoAProducerHSciL1Seeded:timeLayerCluster","hltHgCalLayerClustersFromSoAProducerHSiL1Seeded:timeLayerCluster")
)

from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel

layerClusters = ["hltHgCalLayerClustersFromSoAProducerL1Seeded",
                 "hltHgCalLayerClustersFromSoAProducerHSciL1Seeded",
                 "hltHgCalLayerClustersFromSoAProducerHSiL1Seeded",
                 "hltBarrelLayerClustersEBL1Seeded"]

time_layerclusters = ["hltHgCalLayerClustersFromSoAProducerL1Seeded:timeLayerCluster",
                      "hltHgCalLayerClustersFromSoAProducerHSciL1Seeded:timeLayerCluster",
                      "hltHgCalLayerClustersFromSoAProducerHSiL1Seeded:timeLayerCluster",
                      "hltBarrelLayerClustersEBL1Seeded:timeLayerCluster"]

ticl_barrel.toModify(hltMergeLayerClustersL1Seeded, layerClusters = layerClusters, time_layerclusters = time_layerclusters)
