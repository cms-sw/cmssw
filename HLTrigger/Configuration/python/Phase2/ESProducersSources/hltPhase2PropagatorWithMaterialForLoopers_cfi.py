import FWCore.ParameterSet.Config as cms

from TrackingTools.MaterialEffects.PropagatorsForLoopers_cff import (
    PropagatorWithMaterialForLoopers as _PropagatorWithMaterialForLoopers,
)

hltPhase2PropagatorWithMaterialForLoopers = _PropagatorWithMaterialForLoopers.clone()
