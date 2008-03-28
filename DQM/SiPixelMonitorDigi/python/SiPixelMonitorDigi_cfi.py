import FWCore.ParameterSet.Config as cms

SiPixelDigiSource = cms.EDFilter("SiPixelDigiSource",
    src = cms.InputTag("siPixelDigis"),
    outputFile = cms.string('Pixel_DQM_Digi.root'),
    saveFile = cms.untracked.bool(False)
)


