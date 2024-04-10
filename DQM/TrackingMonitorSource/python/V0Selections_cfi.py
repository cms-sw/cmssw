from DQM.TrackingMonitorSource.v0EventSelector_cfi import *
from DQM.TrackingMonitorSource.v0VertexTrackProducer_cfi import *

KShortEventSelector = v0EventSelector.clone()
LambdaEventSelector = v0EventSelector.clone(
    vertexCompositeCandidates = "generalV0Candidates:Lambda"  
)

KshortTracks = v0VertexTrackProducer.clone()
LambdaTracks = v0VertexTrackProducer.clone(
    vertexCompositeCandidates = "generalV0Candidates:Lambda"
)
