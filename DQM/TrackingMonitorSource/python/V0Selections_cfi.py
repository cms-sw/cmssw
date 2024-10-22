from DQM.TrackingMonitorSource.v0EventSelector_cfi import *
from DQM.TrackingMonitorSource.v0VertexTrackProducer_cfi import *

KShortEventSelector = v0EventSelector.clone(
    vertexCompositeCandidates = "generalV0Candidates:Kshort",
    massMin = 0.47,
    massMax = 0.52,
)

LambdaEventSelector = v0EventSelector.clone(
    vertexCompositeCandidates = "generalV0Candidates:Lambda",
    massMin = 1.11,
    massMax = 1.128
)

KshortTracks = v0VertexTrackProducer.clone(
    vertexCompositeCandidates = "KShortEventSelector"
)
LambdaTracks = v0VertexTrackProducer.clone(
    vertexCompositeCandidates = "LambdaEventSelector"
)
