import FWCore.ParameterSet.Config as cms

# HFEMClusterShape producer
hfEMClusters = cms.EDProducer("HFEMClusterProducer",
    hits = cms.untracked.InputTag("hfreco"),
    minTowerEnergy = cms.double(3.0),
    seedThresholdET = cms.double(5.0)
)


