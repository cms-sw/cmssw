import FWCore.ParameterSet.Config as cms

from CalibTracker.SiPixelESProducers.siPixelQualityESProducer_cfi import siPixelQualityESProducer as _siPixelQualityESProducer

siPixelQualityForRawToDigiESProducer = _siPixelQualityESProducer.clone(
    appendToDataLabel = 'forRawToDigi',
    siPixelQualityFromDbLabel = 'forRawToDigi'
)

# remove siPixelQualityForRawToDigiESProducer when the modifier siPixelQualityRawToDigi is not enabled
def _removeSiPixelQualityForRawToDigiESProducer(process):
    if hasattr(process, 'siPixelQualityForRawToDigiESProducer'):
        del process.siPixelQualityForRawToDigiESProducer

from Configuration.ProcessModifiers.siPixelQualityRawToDigi_cff import siPixelQualityRawToDigi
removeSiPixelQualityForRawToDigiESProducer_ = (~siPixelQualityRawToDigi).makeProcessModifier( _removeSiPixelQualityForRawToDigiESProducer )
