import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripESProducers.SiStripLorentzAngleDepESProducer_cfi import (
    siStripLorentzAngleDepESProducer as _siStripLorentzAngleDepESProducer,
)

hltPhase2siStripLorentzAngleDepESProducer = _siStripLorentzAngleDepESProducer.clone()
