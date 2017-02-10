import FWCore.ParameterSet.Config as cms

import Validation.RecoTrack.PostProcessorTracker_cfi as _PostProcessorTracker_cfi

postProcessorHLTgsfTracking = _PostProcessorTracker_cfi.postProcessorTrack.clone(
    subDirs = ["HLT/EG/Tracking/ValidationWRTtp/*"]
)

postProcessorHLTgsfTrackingSummary = _PostProcessorTracker_cfi.postProcessorTrackSummary.clone(
    subDirs = ["HLT/EG/Tracking/ValidationWRTtp"]
)

postProcessorHLTgsfTrackingSequence = (
    postProcessorHLTgsfTracking +
    postProcessorHLTgsfTrackingSummary
)
