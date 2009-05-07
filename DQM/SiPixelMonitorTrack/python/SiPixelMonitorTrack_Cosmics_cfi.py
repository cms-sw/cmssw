import FWCore.ParameterSet.Config as cms

SiPixelTrackResidualSource_Cosmics = cms.EDFilter("SiPixelTrackResidualSource",
    src = cms.InputTag("siPixelTrackResiduals"),
    clustersrc = cms.InputTag("siPixelClusters"),                            
    debug = cms.untracked.bool(False),                          
    saveFile = cms.untracked.bool(True),
    outputFile = cms.string('Pixel_DQM_TrackResidual.root'),
    TrackCandidateProducer = cms.string('rsTrackCandidatesP5'),
    TrackCandidateLabel = cms.string(''),
    TTRHBuilder = cms.string('WithTrackAngle'),
    Fitter = cms.string('KFFittingSmootherWithOutliersRejectionAndRK'),
    modOn = cms.untracked.bool(False),
    ladOn = cms.untracked.bool(True),
    layOn = cms.untracked.bool(True),
    phiOn = cms.untracked.bool(True),
    ringOn = cms.untracked.bool(True),
    bladeOn = cms.untracked.bool(True),
    diskOn = cms.untracked.bool(True),

    trajectoryInput = cms.InputTag('rsWithMaterialTracksP5')              
)
