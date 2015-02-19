import FWCore.ParameterSet.Config as cms

SiPixelHLTSource = cms.EDAnalyzer("SiPixelHLTSource",
    TopFolderName = cms.string('Pixel'),
    RawInput = cms.InputTag("rawDataCollector"),
    ErrorInput = cms.InputTag("siPixelDigis"),
    outputFile = cms.string('Pixel_DQM_HLT.root'),
    saveFile = cms.untracked.bool(False),
    slowDown = cms.untracked.bool(False),
    DirName = cms.untracked.string('Pixel/FEDIntegrity/')
)


