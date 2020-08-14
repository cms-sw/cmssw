import FWCore.ParameterSet.Config as cms

from TrackingTools.MaterialEffects.MaterialPropagatorParabolicMf_cff import (
    MaterialPropagatorParabolicMF as _MaterialPropagatorParabolicMF,
)

hltPhase2MaterialPropagatorParabolicMF = _MaterialPropagatorParabolicMF.clone()
