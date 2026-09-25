import FWCore.ParameterSet.Config as cms

hltHGCALGPUvsCPUComparisonHists = cms.EDProducer("HGCALGPUvsCPUComparisonHists",
                                                 monitoredLayerClusters = cms.InputTag("hltMergeLayerClusters"),
                                                 referenceLayerClusters = cms.InputTag("hltMergeLayerClustersSerialSync"),
                                                 topFolderName = cms.string('HLT/HeterogeneousComparisons/HGCalMonitoring')
                                                 )
