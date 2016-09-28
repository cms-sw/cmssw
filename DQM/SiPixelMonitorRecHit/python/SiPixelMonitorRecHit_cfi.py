import FWCore.ParameterSet.Config as cms

#
# This object is used to make changes for different running scenarios
#

SiPixelRecHitSource = cms.EDAnalyzer("SiPixelRecHitSource",
    TopFolderName = cms.string('Pixel'),
    src = cms.InputTag("siPixelRecHits"),
    outputFile = cms.string('Pixel_DQM_RecHits.root'),
    saveFile = cms.untracked.bool(False),
    slowDown = cms.untracked.bool(False),
    isPIB = cms.untracked.bool(False),
    modOn = cms.untracked.bool(True),
    twoDimOn = cms.untracked.bool(True),                            
    reducedSet = cms.untracked.bool(True),	
    ladOn = cms.untracked.bool(False),
    layOn = cms.untracked.bool(False),
    phiOn = cms.untracked.bool(False),
    ringOn = cms.untracked.bool(False),
    bladeOn = cms.untracked.bool(False),
    diskOn = cms.untracked.bool(False)
)

# Modify for if the phase 1 pixel detector is active
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify( SiPixelRecHitSource, isUpgrade=cms.untracked.bool(True) )
