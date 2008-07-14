import FWCore.ParameterSet.Config as cms

siPixelFedFillerWordEventNumber = cms.EDFilter("SiPixelFedFillerWordEventNumber",
    InputInstance = cms.untracked.string(''),
    InputLabel = cms.untracked.string('source'),
    SaveFillerWords = cms.bool(False)
)


