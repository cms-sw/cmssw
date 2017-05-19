import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelTrackFitting.pixelTrackFilterByKinematicsDefault_cfi import pixelTrackFilterByKinematicsDefault as _pixelTrackFilterByKinematicsDefault
pixelTrackFilterByKinematics = _pixelTrackFilterByKinematicsDefault.clone()

from Configuration.Eras.Modifier_trackingPhase1PU70_cff import trackingPhase1PU70
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
_cust = dict(
        chi2 = 50.0,
        tipMax = 0.05
)
trackingPhase1PU70.toModify(pixelTrackFilterByKinematics, **_cust)
trackingPhase2PU140.toModify(pixelTrackFilterByKinematics, **_cust)
