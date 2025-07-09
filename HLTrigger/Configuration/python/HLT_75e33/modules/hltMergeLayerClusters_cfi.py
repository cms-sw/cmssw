import FWCore.ParameterSet.Config as cms

# Default lists
default_layerClusters = [
    "hltHgcalLayerClustersEE", 
    "hltHgcalLayerClustersHSci", 
    "hltHgcalLayerClustersHSi"
]

default_time_layerclusters = [
    "hltHgcalLayerClustersEE:timeLayerCluster", 
    "hltHgcalLayerClustersHSci:timeLayerCluster", 
    "hltHgcalLayerClustersHSi:timeLayerCluster"
]

# Define the producer with default lists
hltMergeLayerClusters = cms.EDProducer("MergeClusterProducer",
    layerClusters = cms.VInputTag(*default_layerClusters),
    time_layerclusters = cms.VInputTag(*default_time_layerclusters),
)

# Process modifier: alpaka
from Configuration.ProcessModifiers.alpaka_cff import alpaka
alpaka.toModify(hltMergeLayerClusters,
    layerClusters = cms.VInputTag(
        "hltHgCalLayerClustersFromSoAProducer",
        "hltHgcalLayerClustersHSci",
        "hltHgcalLayerClustersHSi"
    ),
    time_layerclusters = cms.VInputTag(
        "hltHgCalLayerClustersFromSoAProducer:timeLayerCluster",
        "hltHgcalLayerClustersHSci:timeLayerCluster",
        "hltHgcalLayerClustersHSi:timeLayerCluster"
    )
)

# Process modifier: ticl_barrel
from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel

ticl_barrel.toModify(hltMergeLayerClusters,
    layerClusters = cms.VInputTag(
        *default_layerClusters,
        "hltBarrelLayerClustersEB",
        "hltBarrelLayerClustersHB"
    ),
    time_layerclusters = cms.VInputTag(
        *default_time_layerclusters,
        "hltBarrelLayerClustersEB:timeLayerCluster",
        "hltBarrelLayerClustersHB:timeLayerCluster"
    )
)
