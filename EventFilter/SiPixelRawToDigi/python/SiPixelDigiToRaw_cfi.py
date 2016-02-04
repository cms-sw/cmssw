import FWCore.ParameterSet.Config as cms

siPixelRawData = cms.EDProducer("SiPixelDigiToRaw",
    InputLabel = cms.InputTag("simSiPixelDigis")
)



