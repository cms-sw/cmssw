import FWCore.ParameterSet.Config as cms

PixelTripletNoTipGenerator = cms.PSet(

  ComponentName = cms.string('PixelTripletNoTipGenerator'),

  useBending = cms.bool(True),
  useMultScattering = cms.bool(True),

  extraHitRPhitolerance = cms.double(0.032),
  extraHitRZtolerance = cms.double(0.037),

  beamSpot = cms.InputTag("offlineBeamSpot")
)
