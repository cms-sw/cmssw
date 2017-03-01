from RecoTracker.FinalTrackSelectors.TrackCutClassifier_cfi import *
from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
trackingLowPU.toModify(TrackCutClassifier,
    vertices = "pixelVertices"
)
from Configuration.Eras.Modifier_trackingPhase1PU70_cff import trackingPhase1PU70
trackingPhase1PU70.toModify(TrackCutClassifier,
    vertices = "pixelVertices"
)
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toModify(TrackCutClassifier,
    vertices = "pixelVertices"
)
