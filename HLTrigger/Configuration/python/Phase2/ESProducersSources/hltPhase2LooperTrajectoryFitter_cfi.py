import FWCore.ParameterSet.Config as cms

from TrackingTools.TrackFitters.KFTrajectoryFitter_cfi import (
    KFTrajectoryFitter as _KFTrajectoryFitter,
)

hltPhase2LooperTrajectoryFitter = _KFTrajectoryFitter.clone(
    ComponentName="LooperFitter", Propagator="PropagatorWithMaterialForLoopers"
)
