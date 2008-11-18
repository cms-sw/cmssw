import FWCore.ParameterSet.Config as cms

SiPixelMonitorTrackResiduals = cms.EDFilter("SiPixelMonitorTrackResiduals",
    OutputMEsInRootFile = cms.bool(True),
    src = cms.InputTag("ctfWithMaterialTracks"),
    Fitter = cms.string('KFFitter'),
    OutputFileName = cms.string('SiPixelMonitorTrackResiduals.root'),
    TrackCandidateProducer = cms.string('ckfTrackCandidates'),
    TrackCandidateLabel = cms.string(''),
    TTRHBuilder = cms.string('WithTrackAngle')
)


