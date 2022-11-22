import FWCore.ParameterSet.Config as cms

from CalibTracker.SiPixelESProducers.siPixelQualityESProducer_cfi import siPixelQualityESProducer

from Configuration.ProcessModifiers.siPixelQualityRawToDigi_cff import siPixelQualityRawToDigi
siPixelQualityRawToDigi.toModify(siPixelQualityESProducer,
    siPixelQualityLabel_RawToDigi = 'forRawToDigi',
)

