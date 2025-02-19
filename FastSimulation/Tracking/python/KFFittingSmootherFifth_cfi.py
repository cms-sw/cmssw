import FWCore.ParameterSet.Config as cms

KFFittingSmootherFifth = cms.ESProducer(
    "KFFittingSmootherESProducer",
#    EstimateCut = cms.double(20),
    EstimateCut = cms.double(-1),
    Fitter = cms.string('KFFitter'),
#    MinNumberOfHits = cms.int32(7),
    MinNumberOfHits = cms.int32(4),
    Smoother = cms.string('KFSmoother'),
    BreakTrajWith2ConsecutiveMissing = cms.bool(True),
    ComponentName = cms.string('KFFittingSmootherFifth'),
    NoInvalidHitsBeginEnd = cms.bool(True),
    RejectTracks = cms.bool(True),
    LogPixelProbabilityCut = cms.double(-16)
)
