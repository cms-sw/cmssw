import FWCore.ParameterSet.Config as cms

siPixelErrorsDigisToCalibDigis = cms.EDFilter("SiPixelErrorsDigisToCalibDigis",
    saveFile = cms.untracked.bool(True),
    outputFilename = cms.string('myResults.root'),
    SiPixelProducerLabelTag = cms.InputTag("siPixelCalibDigis")
)


