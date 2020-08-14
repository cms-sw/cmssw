import FWCore.ParameterSet.Config as cms

from TrackingTools.TrackFitters.KFTrajectorySmoother_cfi import (
    KFTrajectorySmoother as _KFTrajectorySmoother,
)

hltPhase2LooperTrajectorySmoother = _KFTrajectorySmoother.clone(
    ComponentName="LooperSmoother",
    Propagator="PropagatorWithMaterialForLoopers",
    errorRescaling=10.0,
)
