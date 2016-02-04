import FWCore.ParameterSet.Config as cms

siPixelRawData = cms.EDProducer("SiPixelDigiToRaw",
    Timing = cms.untracked.bool(False),
    InputLabel = cms.InputTag("simSiPixelDigis")
)



