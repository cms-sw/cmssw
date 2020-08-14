import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi import (
    siStripQualityESProducer as _siStripQualityESProducer,
)

hltPhase2siStripQualityESProducer = _siStripQualityESProducer.clone()
