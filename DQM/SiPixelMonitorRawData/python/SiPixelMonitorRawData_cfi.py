import FWCore.ParameterSet.Config as cms

SiPixelRawDataErrorSource = cms.EDAnalyzer("SiPixelRawDataErrorSource",
    TopFolderName = cms.string('Pixel'),
    src = cms.InputTag("siPixelDigis"),
    outputFile = cms.string('Pixel_DQM_Error.root'),
    saveFile = cms.untracked.bool(False),
    isPIB = cms.untracked.bool(False),
    slowDown = cms.untracked.bool(False),
    reducedSet = cms.untracked.bool(False),
    modOn = cms.untracked.bool(True),
    ladOn = cms.untracked.bool(False),
    bladeOn = cms.untracked.bool(False)
)


