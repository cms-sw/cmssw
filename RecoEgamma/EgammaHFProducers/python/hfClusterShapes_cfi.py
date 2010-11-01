import FWCore.ParameterSet.Config as cms

# HFEMClusterShape producer
hfEMClusters = cms.EDProducer("HFEMClusterProducer",
    hits = cms.InputTag("hfreco"),
    minTowerEnergy = cms.double(3.0),
    seedThresholdET = cms.double(5.0),
    maximumSL = cms.double(0.98),
    maximumRenergy  = cms.double(50),                    
    useFlag = cms.bool(False)                              
)


