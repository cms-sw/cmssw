from RecoTracker.PixelTrackFitting.pixelTrackFilterByKinematics_cfi import pixelTrackFilterByKinematics as _pixelTrackFilterByKinematics
pixelTrackFilterByKinematicsForTSGFromL1 = _pixelTrackFilterByKinematics.clone(
    nSigmaInvPtTolerance = 2.0,
    nSigmaTipMaxTolerance = 3.0,
    ptMin = 10.0,
    tipMax = 0.1,
)
