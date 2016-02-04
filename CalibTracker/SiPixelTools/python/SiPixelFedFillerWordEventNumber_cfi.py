import FWCore.ParameterSet.Config as cms

siPixelFedFillerWordEventNumber = cms.EDProducer("SiPixelFedFillerWordEventNumber",
    InputInstance = cms.untracked.string(''),
    InputLabel = cms.untracked.string('source'),
    SaveFillerWords = cms.bool(False)
)


