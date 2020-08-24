import FWCore.ParameterSet.Config as cms

from TrackingTools.TrackFitters.KFTrajectoryFitter_cfi import (
    KFTrajectoryFitter as _KFTrajectoryFitter,
)

import FWCore.ParameterSet.Config as cms

hltPhase2RKTrajectoryFitter = _KFTrajectoryFitter.clone(
    ComponentName="RKFitter", Propagator="RungeKuttaTrackerPropagator"
)
