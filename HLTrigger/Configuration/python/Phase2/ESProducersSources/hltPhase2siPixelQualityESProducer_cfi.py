import FWCore.ParameterSet.Config as cms

from CalibTracker.SiPixelESProducers.SiPixelQualityESProducer_cfi import (
    siPixelQualityESProducer as _siPixelQualityESProducer,
)

hltPhase2siPixelQualityESProducer = _siPixelQualityESProducer.clone(
    ListOfRecordToMerge=cms.VPSet(
        cms.PSet(record=cms.string("SiPixelQualityFromDbRcd"), tag=cms.string("")),
        cms.PSet(record=cms.string("SiPixelDetVOffRcd"), tag=cms.string("")),
    ),
    siPixelQualityLabel="forDigitizer",
)
