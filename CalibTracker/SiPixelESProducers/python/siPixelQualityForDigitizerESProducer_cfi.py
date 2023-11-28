import FWCore.ParameterSet.Config as cms

from CalibTracker.SiPixelESProducers.siPixelQualityESProducer_cfi import siPixelQualityESProducer as _siPixelQualityESProducer

siPixelQualityForDigitizerESProducer = _siPixelQualityESProducer.clone(
    appendToDataLabel = 'forDigitizer',
    siPixelQualityFromDbLabel = 'forDigitizer'
)

# remove siPixelQualityForDigitizerESProducer when the modifier run2_SiPixel_2018 is not enabled
def _removeSiPixelQualityForDigitizerESProducer(process):
    if hasattr(process, 'siPixelQualityForDigitizerESProducer'):
        del process.siPixelQualityForDigitizerESProducer

from Configuration.Eras.Modifier_run2_SiPixel_2018_cff import run2_SiPixel_2018
removeSiPixelQualityForDigitizerESProducer_ = (~run2_SiPixel_2018).makeProcessModifier( _removeSiPixelQualityForDigitizerESProducer )
