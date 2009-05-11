import FWCore.ParameterSet.Config as cms

TkLasBeamFitter = cms.EDProducer(
    "TkLasBeamFitter",
    src = cms.InputTag("LaserAlignment", "tkLaserBeams")
    )

