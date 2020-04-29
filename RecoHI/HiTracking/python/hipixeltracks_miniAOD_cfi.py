import FWCore.ParameterSet.Config as cms

hipixeltracks = cms.EDProducer("HIPixelTracksSlimmer_miniAOD",
  srcTracks = cms.InputTag("hiConformalPixelTracks"),
  srcVertices = cms.InputTag("offlinePrimaryVertices"),
  srcPrimaryVertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
  dzSigCut = cms.double(10.0),
  dxySigCut = cms.double(25.0),
  dzSigHP = cms.double(7.0), #to set HighPurity flag for pixel tracks
  dxySigHP = cms.double(20.0), #to set HighPurity flag for pixel tracks
  ptMax = cms.double(1.0), #max pT for pixel tracks - above this use general tracks instead
  covarianceVersion = cms.int32(0), #so far: 0 is Phase0, 1 is Phase1
  covarianceSchema = cms.int32(520) #use less accurate schema - reduce size of collection by around 20%
)

from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(hipixeltracks, covarianceVersion=1)
