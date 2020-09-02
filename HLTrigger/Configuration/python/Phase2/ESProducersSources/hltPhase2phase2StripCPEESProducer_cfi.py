import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.Phase2TrackerRecHits.Phase2StripCPEESProducer_cfi import (
    phase2StripCPEESProducer as _phase2StripCPEESProducer,
)

hltPhase2StripCPEESProducer = _phase2StripCPEESProducer.clone()
