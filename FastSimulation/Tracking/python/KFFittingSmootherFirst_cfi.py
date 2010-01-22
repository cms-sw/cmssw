import FWCore.ParameterSet.Config as cms

KFFittingSmootherFirst = cms.ESProducer(
    "KFFittingSmootherESProducer",
    EstimateCut = cms.double(20),
    Fitter = cms.string('KFFitter'),
    MinNumberOfHits = cms.int32(7),
    Smoother = cms.string('KFSmoother'),
    BreakTrajWith2ConsecutiveMissing = cms.bool(True),
    ComponentName = cms.string('KFFittingSmootherFirst'),
    NoInvalidHitsBeginEnd = cms.bool(True),
    RejectTracks = cms.bool(True),
    LogPixelProbabilityCut = cms.double(-16)
)
