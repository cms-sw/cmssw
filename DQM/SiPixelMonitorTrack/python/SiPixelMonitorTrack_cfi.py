import FWCore.ParameterSet.Config as cms

#
# This object is used to make changes for different running scenarios
#

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
SiPixelTrackResidualSource = DQMEDAnalyzer('SiPixelTrackResidualSource',
    TopFolderName = cms.string('Pixel'),
    src = cms.InputTag("siPixelTrackResiduals"),
    clustersrc = cms.InputTag("siPixelClusters"),                            
    tracksrc   = cms.InputTag("refittedForPixelDQM"),
    debug = cms.untracked.bool(False),                          
    saveFile = cms.untracked.bool(False),
    outputFile = cms.string('Pixel_DQM_TrackResidual.root'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    Fitter = cms.string('KFFittingSmootherWithOutliersRejectionAndRK'),
    modOn = cms.untracked.bool(False),
    reducedSet = cms.untracked.bool(True),
    ladOn = cms.untracked.bool(True),
    layOn = cms.untracked.bool(True),
    phiOn = cms.untracked.bool(True),
    ringOn = cms.untracked.bool(True),
    bladeOn = cms.untracked.bool(True),
    diskOn = cms.untracked.bool(True),
    PtMinRes = cms.untracked.double(4.0),
    vtxsrc= cms.untracked.string("offlinePrimaryVertices"),

    trajectoryInput = cms.InputTag('refittedForPixelDQM'),              
    digisrc = cms.InputTag("siPixelDigis") 
)

# Modify for if the phase 1 pixel detector is active
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify( SiPixelTrackResidualSource, isUpgrade=cms.untracked.bool(True) )
