import FWCore.ParameterSet.Config as cms

# as in the e/gamma L1-seeded chain, the EE layer clusters come from the CPU producer also under alpaka
hltMergeLayerClustersMuonSeeded = cms.EDProducer("MergeClusterProducer",
    layerClusters = cms.VInputTag("hltHgcalLayerClustersEEMuonSeeded", "hltHgcalLayerClustersHSciMuonSeeded", "hltHgcalLayerClustersHSiMuonSeeded"),
    time_layerclusters = cms.VInputTag("hltHgcalLayerClustersEEMuonSeeded:timeLayerCluster", "hltHgcalLayerClustersHSciMuonSeeded:timeLayerCluster", "hltHgcalLayerClustersHSiMuonSeeded:timeLayerCluster")
)

# barrel hadronic layer clusters are taken from the full HCAL reconstruction
from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel
ticl_barrel.toModify(hltMergeLayerClustersMuonSeeded,
    layerClusters = ["hltHgcalLayerClustersEEMuonSeeded",
                     "hltHgcalLayerClustersHSciMuonSeeded",
                     "hltHgcalLayerClustersHSiMuonSeeded",
                     "hltBarrelLayerClustersEBMuonSeeded",
                     "hltBarrelLayerClustersHB"],
    time_layerclusters = ["hltHgcalLayerClustersEEMuonSeeded:timeLayerCluster",
                          "hltHgcalLayerClustersHSciMuonSeeded:timeLayerCluster",
                          "hltHgcalLayerClustersHSiMuonSeeded:timeLayerCluster",
                          "hltBarrelLayerClustersEBMuonSeeded:timeLayerCluster",
                          "hltBarrelLayerClustersHB:timeLayerCluster"]
)
