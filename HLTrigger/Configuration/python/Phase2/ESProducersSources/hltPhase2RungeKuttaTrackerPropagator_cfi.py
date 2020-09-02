import FWCore.ParameterSet.Config as cms

from TrackingTools.MaterialEffects.RungeKuttaTrackerPropagator_cfi import (
    RungeKuttaTrackerPropagator as _RungeKuttaTrackerPropagator,
)

hltPhase2RungeKuttaTrackerPropagator = _RungeKuttaTrackerPropagator.clone()
