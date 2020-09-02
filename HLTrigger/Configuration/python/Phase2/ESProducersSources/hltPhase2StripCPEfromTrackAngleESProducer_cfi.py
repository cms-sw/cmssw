import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import (
    StripCPEfromTrackAngleESProducer as _StripCPEfromTrackAngleESProducer,
)

hltPhase2StripCPEfromTrackAngleESProducer = _StripCPEfromTrackAngleESProducer.clone()
