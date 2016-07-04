from Configuration.StandardSequences.Eras import eras
from RecoTracker.FinalTrackSelectors.TrackCutClassifier_cfi import *
eras.trackingLowPU.toModify(TrackCutClassifier,
    vertices = "pixelVertices"
)
eras.trackingPhase1PU70.toModify(TrackCutClassifier,
    vertices = "pixelVertices"
)
eras.trackingPhase2PU140.toModify(TrackCutClassifier,
    vertices = "pixelVertices"
)
