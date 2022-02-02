import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.patTracksToPackedCandidates_cfi import patTracksToPackedCandidates

hiPixelTracks = patTracksToPackedCandidates.clone(
  dxySigCut = 3.0,
  dzSigCut = 3.0,
  dxySigHP = 2.0,
  dzSigHP = 2.0
)

from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(hiPixelTracks, covarianceVersion=1)
