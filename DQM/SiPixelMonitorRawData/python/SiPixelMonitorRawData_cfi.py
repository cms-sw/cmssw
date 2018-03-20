import FWCore.ParameterSet.Config as cms

#
# This object is used to make changes for different running scenarios
#

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
SiPixelRawDataErrorSource = DQMEDAnalyzer('SiPixelRawDataErrorSource',
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

# Modify for if the phase 1 pixel detector is active
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify( SiPixelRawDataErrorSource, isUpgrade=cms.untracked.bool(True) )
