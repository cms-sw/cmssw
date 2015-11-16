import FWCore.ParameterSet.Config as cms

#
# This object is used to make changes for different running scenarios
#
from Configuration.StandardSequences.Eras import eras

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

# Modify for if the phase 1 pixel detector is active
eras.phase1Pixel.toModify( SiPixelRawDataErrorSource, isUpgrade=cms.untracked.bool(True) )
