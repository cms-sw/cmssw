import FWCore.ParameterSet.Config as cms

tobTecStepFitterSmootherForLoopers = cms.ESProducer("KFFittingSmootherESProducer",
    BreakTrajWith2ConsecutiveMissing = cms.bool(True),
    ComponentName = cms.string('tobTecStepFitterSmootherForLoopers'),
    EstimateCut = cms.double(30),
    Fitter = cms.string('tobTecStepRKFitterForLoopers'),
    LogPixelProbabilityCut = cms.double(0),
    MaxFractionOutliers = cms.double(0.3),
    MaxNumberOfOutliers = cms.int32(3),
    MinDof = cms.int32(2),
    MinNumberOfHits = cms.int32(7),
    NoInvalidHitsBeginEnd = cms.bool(True),
    NoOutliersBeginEnd = cms.bool(False),
    RejectTracks = cms.bool(True),
    Smoother = cms.string('tobTecStepRKSmootherForLoopers'),
    appendToDataLabel = cms.string('')
)
