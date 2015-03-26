import FWCore.ParameterSet.Config as cms

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

    trajectoryInput = cms.InputTag('generalTracks')              
)
