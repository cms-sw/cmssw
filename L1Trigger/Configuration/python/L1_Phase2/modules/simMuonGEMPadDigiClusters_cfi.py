import FWCore.ParameterSet.Config as cms

simMuonGEMPadDigiClusters = cms.EDProducer("GEMPadDigiClusterProducer",
    InputCollection = cms.InputTag("simMuonGEMPadDigis"),
    maxClusterSize = cms.uint32(8),
    maxClustersOHGE11 = cms.uint32(8),
    maxClustersOHGE21 = cms.uint32(8),
    maxClustersPartitionGE11 = cms.uint32(4),
    maxClustersPartitionGE21 = cms.uint32(4),
    mightGet = cms.optional.untracked.vstring,
    nOHGE11 = cms.uint32(1),
    nOHGE21 = cms.uint32(4),
    nPartitionsGE11 = cms.uint32(4),
    nPartitionsGE21 = cms.uint32(4),
    sendOverflowClusters = cms.bool(False)
)
