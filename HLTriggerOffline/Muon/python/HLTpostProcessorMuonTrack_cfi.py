import FWCore.ParameterSet.Config as cms

import Validation.RecoTrack.PostProcessorTracker_cfi as _PostProcessorTracker_cfi

postProcessorHLTmuonTracking = _PostProcessorTracker_cfi.postProcessorTrack.clone(
    subDirs = ["HLT/Muon/Tracking/ValidationWRTtp/*"]
)

postProcessorHLTmuonTrackingSummary = _PostProcessorTracker_cfi.postProcessorTrackSummary.clone(
    subDirs = ["HLT/Muon/Tracking/ValidationWRTtp"]
)

postProcessorHLTmuonTrackingSequence = (
    postProcessorHLTmuonTracking +
    postProcessorHLTmuonTrackingSummary
)
