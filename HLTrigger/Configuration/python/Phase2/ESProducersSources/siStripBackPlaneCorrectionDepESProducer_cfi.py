import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripESProducers.SiStripBackPlaneCorrectionDepESProducer_cfi import (
    siStripBackPlaneCorrectionDepESProducer as _siStripBackPlaneCorrectionDepESProducer,
)

hltPhase2siStripBackPlaneCorrectionDepESProducer = (
    _siStripBackPlaneCorrectionDepESProducer.clone()
)
