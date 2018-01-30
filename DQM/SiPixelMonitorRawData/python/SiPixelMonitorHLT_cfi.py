import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
SiPixelHLTSource = DQMEDAnalyzer('SiPixelHLTSource',
    DirName = cms.untracked.string('Pixel/FEDIntegrity/'),
    RawInput = cms.InputTag("rawDataCollector"),
    ErrorInput = cms.InputTag("siPixelDigis"),
    outputFile = cms.string('Pixel_DQM_HLT.root'),
    saveFile = cms.untracked.bool(False),
    slowDown = cms.untracked.bool(False)
)


