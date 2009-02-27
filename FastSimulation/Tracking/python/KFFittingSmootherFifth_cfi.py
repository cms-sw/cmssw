import FWCore.ParameterSet.Config as cms

KFFittingSmootherFifth = cms.ESProducer(
    "KFFittingSmootherESProducer",
    EstimateCut = cms.double(20.0),
    Fitter = cms.string('KFFitter'),
    MinNumberOfHits = cms.int32(6),
    Smoother = cms.string('KFSmoother'),
    BreakTrajWith2ConsecutiveMissing = cms.bool(True),
    ComponentName = cms.string('KFFittingSmootherFifth'),
    NoInvalidHitsBeginEnd = cms.bool(True),
    RejectTracks = cms.bool(True)
)
