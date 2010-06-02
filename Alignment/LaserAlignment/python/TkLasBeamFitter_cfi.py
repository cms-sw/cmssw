import FWCore.ParameterSet.Config as cms

TkLasBeamFitter = cms.EDProducer(
    "TkLasBeamFitter",
    src = cms.InputTag("LaserAlignment", "tkLaserBeams"),
    # Fit Beam Splitters? If not, preset values are taken
    fitBeamSplitters = cms.bool( True ),
    # AT fit params: 6 is recommended; only other valid values are '3' or '5'
    # '3': slope, offset, Beam Splitters are fitted
    # '5': rotations of both ATs are fitted as well
    # '6': relative shift of ATs in phi-z-plane is fitted
    numberOfFittedAtParameters = cms.uint32( 6 )
)

