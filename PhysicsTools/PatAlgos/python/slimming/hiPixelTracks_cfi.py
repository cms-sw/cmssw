import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.patTracksToPackedCandidates_cfi import patTracksToPackedCandidates

hiPixelTracks = patTracksToPackedCandidates.clone()
hiPixelTracks.dxySigCut = 3.
hiPixelTracks.dzSigCut = 3.
hiPixelTracks.dxySigHP = 2.
hiPixelTracks.dzSigHP = 2.
hiPixelTracks.ptMin = 0.2

from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(hiPixelTracks, covarianceVersion=1)
