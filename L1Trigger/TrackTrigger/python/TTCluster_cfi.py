import FWCore.ParameterSet.Config as cms

TTClustersFromPixelDigis = cms.EDProducer("TTClusterBuilder_PixelDigi_",
    rawHits = cms.VInputTag(cms.InputTag("simSiPixelDigis")),
    simTrackHits = cms.InputTag("g4SimHits"),
    ADCThreshold = cms.uint32(30),
    storeLocalCoord = cms.bool(True), # if True, local coordinates (row and col)
                                      # of each hit are stored in the cluster object
)

