import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.Phase2TrackerRecHits.Phase2TrackerRecHits_cfi import siPhase2RecHits as _siPhase2RecHits
hltSiPhase2RecHits = _siPhase2RecHits.clone( src = "hltSiPhase2Clusters" )

