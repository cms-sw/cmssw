import FWCore.ParameterSet.Config as cms

siPixelQualityESProducer = cms.ESProducer("SiPixelQualityESProducer",
    siPixelQualityLabel = cms.string(""),
    siPixelQualityLabel_RawToDigi = cms.string("")
)

from Configuration.ProcessModifiers.siPixelQualityRawToDigi_cff import siPixelQualityRawToDigi
siPixelQualityRawToDigi.toModify(siPixelQualityESProducer,
    siPixelQualityLabel_RawToDigi = 'forRawToDigi',
)

