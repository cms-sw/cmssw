import FWCore.ParameterSet.Config as cms

SiPixelTrackResidualSource = cms.EDFilter("SiPixelTrackResidualSource",
    src = cms.InputTag("siPixelTrackResiduals"),
    debug = cms.untracked.bool(False),
    saveFile = cms.untracked.bool(True),
    outputFile = cms.string('Pixel_DQM_TrackResidual.root'),
    TrackCandidateProducer = cms.string('newTrackCandidateMaker'),
    TrackCandidateLabel = cms.string(''),
    TTRHBuilder = cms.string('WithTrackAngle'),
    Fitter = cms.string('KFFittingSmootherWithOutliersRejectionAndRK')
)
