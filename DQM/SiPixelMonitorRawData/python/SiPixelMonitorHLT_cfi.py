import FWCore.ParameterSet.Config as cms

SiPixelHLTSource = cms.EDFilter("SiPixelHLTSource",
    RawInput = cms.InputTag("source"),
    ErrorInput = cms.InputTag("siPixelDigis"),
    outputFile = cms.string('Pixel_DQM_HLT.root'),
    saveFile = cms.untracked.bool(False),
    slowDown = cms.untracked.bool(False),
)


