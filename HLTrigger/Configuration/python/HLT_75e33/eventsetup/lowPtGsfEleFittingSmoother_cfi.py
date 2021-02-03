import FWCore.ParameterSet.Config as cms

lowPtGsfEleFittingSmoother = cms.ESProducer("KFFittingSmootherESProducer",
    BreakTrajWith2ConsecutiveMissing = cms.bool(True),
    ComponentName = cms.string('lowPtGsfEleFittingSmoother'),
    EstimateCut = cms.double(-1),
    Fitter = cms.string('GsfTrajectoryFitter'),
    LogPixelProbabilityCut = cms.double(0),
    MaxFractionOutliers = cms.double(0.3),
    MaxNumberOfOutliers = cms.int32(3),
    MinDof = cms.int32(2),
    MinNumberOfHits = cms.int32(2),
    NoInvalidHitsBeginEnd = cms.bool(True),
    NoOutliersBeginEnd = cms.bool(False),
    RejectTracks = cms.bool(True),
    Smoother = cms.string('GsfTrajectorySmoother'),
    appendToDataLabel = cms.string('')
)
