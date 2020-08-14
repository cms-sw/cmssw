import FWCore.ParameterSet.Config as cms

from RecoTracker.FinalTrackSelectors.trackAlgoPriorityOrderDefault_cfi import (
    trackAlgoPriorityOrderDefault as _trackAlgoPriorityOrderDefault,
)

hltPhase2trackAlgoPriorityOrder = _trackAlgoPriorityOrderDefault.clone(
    algoOrder=["hltIter0", "initialStep", "highPtTripletStep"]
)
