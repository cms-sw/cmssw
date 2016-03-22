from Configuration.StandardSequences.Eras import eras
from RecoTracker.FinalTrackSelectors.TrackCutClassifier_cfi import *
eras.trackingPhase1.toModify(TrackCutClassifier,
    vertices = "pixelVertices"
)
