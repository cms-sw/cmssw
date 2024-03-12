import FWCore.ParameterSet.Config as cms

siPixelErrorsDigisToCalibDigis = cms.EDAnalyzer("SiPixelErrorsDigisToCalibDigis",
    saveFile = cms.untracked.bool(True),
    outputFilename = cms.string('myResults.root'),
    SiPixelProducerLabelTag = cms.InputTag("siPixelCalibDigis")
)


# foo bar baz
# 0ruaOhVyPc8JX
# uEJ50SO1LCe1r
