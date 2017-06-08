import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelTrackFitting.pixelTrackFilterByKinematicsDefault_cfi import pixelTrackFilterByKinematicsDefault as _pixelTrackFilterByKinematicsDefault
pixelTrackFilterByKinematics = _pixelTrackFilterByKinematicsDefault.clone()

from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toModify(pixelTrackFilterByKinematics,
    chi2 = 50.0,
    tipMax = 0.05
)
