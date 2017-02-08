import FWCore.ParameterSet.Config as cms

ctppsDiamondLocalTracks = cms.EDProducer("CTPPSDiamondLocalTrackFitter",
    verbosity = cms.int32(0),
    recHitsTag = cms.InputTag("ctppsDiamondRecHits"),
    trackingAlgorithmParams = cms.PSet(
        threshold = cms.double(2.0),
        resolution = cms.double(0.01), # in mm
        sigma = cms.double(0.0),
    ),
)
