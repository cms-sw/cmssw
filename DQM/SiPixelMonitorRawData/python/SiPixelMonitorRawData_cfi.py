import FWCore.ParameterSet.Config as cms

SiPixelRawDataErrorSource = cms.EDFilter("SiPixelRawDataErrorSource",
    src = cms.InputTag("siPixelDigis"),
    outputFile = cms.string('Pixel_DQM_Error.root'),
    saveFile = cms.untracked.bool(False)
)


