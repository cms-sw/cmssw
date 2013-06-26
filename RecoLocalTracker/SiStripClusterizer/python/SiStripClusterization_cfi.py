import FWCore.ParameterSet.Config as cms

SiStripClusterization = cms.PSet(
    ClusterizerAlgorithm = cms.untracked.string('ThreeThreshold'),
    ChannelThreshold     = cms.untracked.double(2.0),
    SeedThreshold        = cms.untracked.double(3.0),
    MaxHolesInCluster    = cms.untracked.uint32(0),
    ClusterThreshold     = cms.untracked.double(5.0)
)
