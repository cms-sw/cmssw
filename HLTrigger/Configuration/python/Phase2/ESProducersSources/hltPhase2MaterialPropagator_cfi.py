import FWCore.ParameterSet.Config as cms

from TrackingTools.MaterialEffects.MaterialPropagator_cfi import (
    MaterialPropagator as _MaterialPropagator,
)

hltPhase2MaterialPropagator = _MaterialPropagator.clone()
