import FWCore.ParameterSet.Config as cms

from TrackingTools.MaterialEffects.MaterialPropagatorParabolicMf_cff import (
    OppositeMaterialPropagatorParabolicMF as _OppositeMaterialPropagatorParabolicMF,
)

hltPhase2OppositeMaterialPropagatorParabolicMF = (
    _OppositeMaterialPropagatorParabolicMF.clone()
)
