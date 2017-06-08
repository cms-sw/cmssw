import FWCore.ParameterSet.Config as cms
from RecoPixelVertexing.PixelTrackFitting.pixelTracksDefault_cfi import pixelTracksDefault as _pixelTracksDefault
pixelTracks = _pixelTracksDefault.clone()

from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
_SeedingHitSets = dict(SeedingHitSets = "pixelTracksHitQuadruplets")
trackingPhase2PU140.toModify(pixelTracks, **_SeedingHitSets)
