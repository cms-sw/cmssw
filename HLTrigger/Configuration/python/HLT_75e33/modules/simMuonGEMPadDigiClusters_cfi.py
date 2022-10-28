import FWCore.ParameterSet.Config as cms

simMuonGEMPadDigiClusters = cms.EDProducer("GEMPadDigiClusterProducer",
    InputCollection = cms.InputTag("simMuonGEMPadDigis"),
    maxClusterSize = cms.uint32(8),
    maxClustersOHGE11 = cms.uint32(4),
    maxClustersOHGE21 = cms.uint32(5),
    mightGet = cms.optional.untracked.vstring,
    nOHGE11 = cms.uint32(2),
    nOHGE21 = cms.uint32(4)
)
