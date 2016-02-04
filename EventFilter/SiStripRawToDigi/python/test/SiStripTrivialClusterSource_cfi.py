import FWCore.ParameterSet.Config as cms

ClusterSource = cms.EDFilter(
    "SiStripTrivialClusterSource",
    MinCluster = cms.untracked.uint32(4),
    MaxCluster = cms.untracked.uint32(4),
    MinOccupancy = cms.untracked.double(0.0056),
    MaxOccupancy = cms.untracked.double(0.0056),
    Separation = cms.untracked.uint32(2),
    )

