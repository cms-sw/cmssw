import FWCore.ParameterSet.Config as cms

PixelTripletNoTipGenerator = cms.PSet(

  ComponentName = cms.string('PixelTripletNoTipGenerator'),

  useBending = cms.bool(True),
  useMultScattering = cms.bool(True),

  extraHitRPhitolerance = cms.double(0.000),
  extraHitRZtolerance = cms.double(0.030),
  nSigma = cms.double(3.0),

  beamSpot = cms.InputTag("offlineBeamSpot")
)
