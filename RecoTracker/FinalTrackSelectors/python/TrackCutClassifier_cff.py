from RecoTracker.FinalTrackSelectors.TrackCutClassifier_cfi import *
from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
trackingLowPU.toModify(TrackCutClassifier,
    vertices = "pixelVertices"
)
