import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import (
    SiStripRecHitMatcherESProducer as _SiStripRecHitMatcherESProducer,
)

hltPhase2SiStripRecHitMatcherESProducer = _SiStripRecHitMatcherESProducer.clone()
