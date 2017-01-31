import FWCore.ParameterSet.Config as cms

ctppsDiamondLocalTrack = cms.EDProducer("CTPPSDiamondLocalTrackFitter",
    verbosity = cms.int32(0),
    recHitsTag = cms.InputTag("ctppsDiamondRecHit"),
    trackingAlgorithmParams = cms.PSet(
        threshold = cms.double(2.0),
        resolution = cms.double(0.01),
        sigma = cms.double(0.0),
    ),
)
