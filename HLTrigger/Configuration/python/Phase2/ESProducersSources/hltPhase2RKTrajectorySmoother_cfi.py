import FWCore.ParameterSet.Config as cms

from TrackingTools.TrackFitters.KFTrajectorySmoother_cfi import (
    KFTrajectorySmoother as _KFTrajectorySmoother,
)

hltPhase2RKTrajectorySmoother = _KFTrajectorySmoother.clone(
    ComponentName="RKSmoother", Propagator="RungeKuttaTrackerPropagator"
)
