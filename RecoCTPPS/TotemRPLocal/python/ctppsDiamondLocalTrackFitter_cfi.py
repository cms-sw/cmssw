import FWCore.ParameterSet.Config as cms

ctppsDiamondLocalTrack = cms.EDProducer("CTPPSDiamondLocalTrackFitter",
    verbosity = cms.int32(0),
    recHitLabel = cms.InputTag("ctppsDiamondRecHit"),
)
