import FWCore.ParameterSet.Config as cms

SiPixelMonitorTrackResiduals = cms.EDFilter("SiPixelMonitorTrackResiduals",
    src = cms.InputTag("siPixelTrackResiduals"),
    debug = cms.bool(True),
    saveFile = cms.bool(True),
    outputFilename = cms.string('Pixel_DQM_TrackResidual.root'),
    TrackCandidateProducer = cms.string('newTrackCandidateMaker'),
    TrackCandidateLabel = cms.string(''),
    TTRHBuilder = cms.string('WithTrackAngle'),
    Fitter = cms.string('FittingSmootherWithOutlierRejection')
)
