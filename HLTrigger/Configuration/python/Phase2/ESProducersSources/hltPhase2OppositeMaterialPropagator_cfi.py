import FWCore.ParameterSet.Config as cms

from TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi import (
    OppositeMaterialPropagator as _OppositeMaterialPropagator,
)

hltPhase2OppositeMaterialPropagator = _OppositeMaterialPropagator.clone()
