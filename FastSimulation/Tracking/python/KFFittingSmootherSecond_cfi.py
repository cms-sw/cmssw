import FWCore.ParameterSet.Config as cms

KFFittingSmootherSecond = cms.ESProducer(
    "KFFittingSmootherESProducer",
    EstimateCut = cms.double(10.0),
    Fitter = cms.string('KFFitter'),
    MinNumberOfHits = cms.int32(3),
    Smoother = cms.string('KFSmoother'),
    BreakTrajWith2ConsecutiveMissing = cms.bool(True),
    ComponentName = cms.string('KFFittingSmootherSecond'),
    NoInvalidHitsBeginEnd = cms.bool(True),
    RejectTracks = cms.bool(True),
    LogPixelProbabilityCut = cms.double(-16)
)
