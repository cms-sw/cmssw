import FWCore.ParameterSet.Config as cms

hltHgcalMergeLayerClusters = cms.EDProducer("MergeClusterProducer",
    layerClustersEE = cms.InputTag("hltHgcalLayerClustersEE"),
    layerClustersHSci = cms.InputTag("hltHgcalLayerClustersHSci"),
    layerClustersHSi = cms.InputTag("hltHgcalLayerClustersHSi"),
    layerClustersEB = cms.InputTag("hltBarrelLayerClustersEB"),
    layerClustersHB = cms.InputTag("hltBarrelLayerClustersHB"),
    mightGet = cms.optional.untracked.vstring,
    timeClname = cms.string('timeLayerCluster'),
    time_layerclustersEE = cms.InputTag("hltHgcalLayerClustersEE","timeLayerCluster"),
    time_layerclustersHSci = cms.InputTag("hltHgcalLayerClustersHSci","timeLayerCluster"),
    time_layerclustersHSi = cms.InputTag("hltHgcalLayerClustersHSi","timeLayerCluster"),
    time_layerclustersEB = cms.InputTag("hltBarrelLayerClustersEB","timeLayerCluster"),
    time_layerclustersHB = cms.InputTag("hltBarrelLayerClustersHB","timeLayerCluster")
)

from Configuration.ProcessModifiers.alpaka_cff import alpaka
alpaka.toModify(hltHgcalMergeLayerClusters,
                layerClustersEE = cms.InputTag("hltHgCalLayerClustersFromSoAProducer"),
                time_layerclustersEE = cms.InputTag("hltHgCalLayerClustersFromSoAProducer", "timeLayerCluster"))
