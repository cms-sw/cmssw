import FWCore.ParameterSet.Config as cms

#
# This object is used to make changes for different running scenarios
#
from Configuration.StandardSequences.Eras import eras

SiPixelTrackResidualSource = cms.EDAnalyzer("SiPixelTrackResidualSource",
    TopFolderName = cms.string('Pixel'),
    src = cms.InputTag("siPixelTrackResiduals"),
    clustersrc = cms.InputTag("siPixelClusters"),                            
    tracksrc   = cms.InputTag("generalTracks"),
    debug = cms.untracked.bool(False),                          
    saveFile = cms.untracked.bool(False),
    outputFile = cms.string('Pixel_DQM_TrackResidual.root'),
    TrackCandidateProducer = cms.string('initialStepTrackCandidates'),
    TrackCandidateLabel = cms.string(''),
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

    trajectoryInput = cms.InputTag('generalTracks'),              
    digisrc = cms.InputTag("siPixelDigis") 
)

# Modify for if the phase 1 pixel detector is active
eras.phase1Pixel.toModify( SiPixelTrackResidualSource, isUpgrade=cms.untracked.bool(True) )
