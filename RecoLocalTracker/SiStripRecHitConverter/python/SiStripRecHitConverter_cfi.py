import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripRecHitConverter.siStripRecHitConverter_cfi import siStripRecHitConverter as _siStripRecHitConverter

siStripMatchedRecHits = _siStripRecHitConverter.clone()
