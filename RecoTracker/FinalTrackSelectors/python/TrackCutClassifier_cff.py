from Configuration.StandardSequences.Eras import eras
from RecoTracker.FinalTrackSelectors.TrackCutClassifier_cfi import *
eras.trackingLowPU.toModify(TrackCutClassifier,
    vertices = "pixelVertices"
)
