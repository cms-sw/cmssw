import FWCore.ParameterSet.Config as cms

KFFittingSmootherFourth = cms.ESProducer(
    "KFFittingSmootherESProducer",
    EstimateCut = cms.double(-1),
    Fitter = cms.string('KFFitter'),
    MinNumberOfHits = cms.int32(3),
    Smoother = cms.string('KFSmoother'),
    BreakTrajWith2ConsecutiveMissing = cms.bool(True),
    ComponentName = cms.string('KFFittingSmootherFourth'),
    NoInvalidHitsBeginEnd = cms.bool(True),
    RejectTracks = cms.bool(True),
    LogPixelProbabilityCut = cms.double(-16)
)
