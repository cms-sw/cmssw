import FWCore.ParameterSet.Config as cms

mtdMergedClusters = cms.EDProducer("MTDMergedClusterProducer",
    btlBarrel = cms.InputTag("mtdClusters", "FTLBarrel"),
    etlEndcap = cms.InputTag("mtdClusters", "FTLEndcap"),
    timeThreshold = cms.double(10.0),
    energyThreshold = cms.double(0.0),
    btlMergedClusterInstance = cms.string("FTLBarrel"),
    etlMergedClusterInstance = cms.string("FTLEndcap")
)