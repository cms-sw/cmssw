import FWCore.ParameterSet.Config as cms

pixelToLNKAssociateFromAscii = cms.ESProducer("PixelToLNKAssociateFromAsciiESProducer",
  fileName =  cms.string('pixelToLNK.ascii')
)
