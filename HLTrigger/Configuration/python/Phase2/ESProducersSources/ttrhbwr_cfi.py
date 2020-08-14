import FWCore.ParameterSet.Config as cms

from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import (
    ttrhbwr as _ttrhbwr,
)

hltPhase2ttrhbwr = _ttrhbwr.clone(Phase2StripCPE=cms.string("Phase2StripCPE"))
