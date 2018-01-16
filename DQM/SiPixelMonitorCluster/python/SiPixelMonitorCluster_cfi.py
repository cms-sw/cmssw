import FWCore.ParameterSet.Config as cms

#
# This object is used to make changes for different running scenarios
#

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
SiPixelClusterSource = DQMEDAnalyzer('SiPixelClusterSource',
    TopFolderName = cms.string('Pixel'),
    src = cms.InputTag("siPixelClusters"),
    digisrc = cms.InputTag("siPixelDigis"),
    outputFile = cms.string('Pixel_DQM_Cluster.root'),
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
    diskOn = cms.untracked.bool(False),
    smileyOn = cms.untracked.bool(True),
    bigEventSize = cms.untracked.int32(100)
)

# Modify for if the phase 1 pixel detector is active
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify( SiPixelClusterSource, isUpgrade=cms.untracked.bool(True) )
