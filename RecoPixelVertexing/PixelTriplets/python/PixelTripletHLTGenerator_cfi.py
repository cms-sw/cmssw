import FWCore.ParameterSet.Config as cms

GeneratorPSet = cms.PSet(
    useBending = cms.bool(True),
    useFixedPreFiltering = cms.bool(False),
    ComponentName = cms.string('PixelTripletHLTGenerator'),
    extraHitRPhitolerance = cms.double(0.06),
    useMultScattering = cms.bool(True),
    phiPreFiltering = cms.double(0.3), ## can be removed if !useFixedPreFiltering

    extraHitRZtolerance = cms.double(0.06)
)

