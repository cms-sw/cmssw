import FWCore.ParameterSet.Config as cms

TTClustersFromPhase2TrackerDigis = cms.EDProducer("TTClusterBuilder_Phase2TrackerDigi_",
    ADCThreshold = cms.uint32(30),
    rawHits = cms.VInputTag(cms.InputTag("mix","Tracker")),
    storeLocalCoord = cms.bool(True)
)
