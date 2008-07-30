import FWCore.ParameterSet.Config as cms

KFFittingSmootherWithOutliersRejectionAndRK = cms.ESProducer(
    "KFFittingSmootherESProducer",
    EstimateCut = cms.double(20.0),
    Fitter = cms.string('KFFitter'),
    MinNumberOfHits = cms.int32(3),
    Smoother = cms.string('KFSmoother'),
    BreakTrajWith2ConsecutiveMissing = cms.bool(True),
    ComponentName = cms.string('KFFittingSmootherWithOutliersRejection'),
    NoInvalidHitsBeginEnd = cms.bool(True),
    RejectTracks = cms.bool(True)
)
