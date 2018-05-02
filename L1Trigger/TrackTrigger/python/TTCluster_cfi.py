import FWCore.ParameterSet.Config as cms

TTClustersFromPhase2TrackerDigis = cms.EDProducer("TTClusterBuilder_Phase2TrackerDigi_",
    rawHits = cms.VInputTag(cms.InputTag("mix","Tracker")),
    ADCThreshold = cms.uint32(30),
    storeLocalCoord = cms.bool(True), # if True, local coordinates (row and col)
                                      # of each hit are stored in the cluster object
)

