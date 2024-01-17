import FWCore.ParameterSet.Config as cms

PixelTripletHLTGenerator = cms.PSet(
    ComponentName = cms.string('PixelTripletHLTGenerator'),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    extraHitRPhitolerance = cms.double(0.016),
    extraHitRZtolerance = cms.double(0.02),
    maxElement = cms.uint32(100000),
    phiPreFiltering = cms.double(0.3),
    useBending = cms.bool(True),
    useFixedPreFiltering = cms.bool(False),
    useMultScattering = cms.bool(True)
)