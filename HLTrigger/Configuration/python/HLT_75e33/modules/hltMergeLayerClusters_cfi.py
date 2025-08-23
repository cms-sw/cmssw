import FWCore.ParameterSet.Config as cms

ceh_layerClusters = [
    "hltHgcalLayerClustersHSci", 
    "hltHgcalLayerClustersHSi"
]
ceh_time_layerClusters = [x + ":timeLayerCluster" for x in ceh_layerClusters]

barrel_layerClusters = [
    "hltBarrelLayerClustersEB",
    "hltBarrelLayerClustersHB"
]
barrel_time_layerClusters = [x + ":timeLayerCluster" for x in barrel_layerClusters]

# Define the producer with ceh lists
hltMergeLayerClusters = cms.EDProducer("MergeClusterProducer",
    layerClusters = cms.VInputTag("hltHgcalLayerClustersEE", *ceh_layerClusters),
    time_layerclusters = cms.VInputTag("hltHgcalLayerClustersEE:timeLayerCluster", *ceh_time_layerClusters),
)

# Process modifiers: ticl_barrel and alpaka
from Configuration.ProcessModifiers.alpaka_cff import alpaka
from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel

(alpaka & ~ticl_barrel).toModify(hltMergeLayerClusters,
    layerClusters = ["hltHgCalLayerClustersFromSoAProducer", *ceh_layerClusters],
    time_layerclusters = ["hltHgCalLayerClustersFromSoAProducer:timeLayerCluster", *ceh_time_layerClusters]
)

(ticl_barrel & ~alpaka).toModify(hltMergeLayerClusters,
    layerClusters = ["hltHgcalLayerClustersEE", *ceh_layerClusters, *barrel_layerClusters],
    time_layerclusters = ["hltHgcalLayerClustersEE:timeLayerCluster", *ceh_time_layerClusters, *barrel_time_layerClusters]
)

(ticl_barrel & alpaka).toModify(hltMergeLayerClusters,
    layerClusters = ["hltHgCalLayerClustersFromSoAProducer", *ceh_layerClusters, *barrel_layerClusters],
    time_layerclusters = ["hltHgCalLayerClustersFromSoAProducer:timeLayerCluster", *ceh_time_layerClusters, *barrel_time_layerClusters]
)
