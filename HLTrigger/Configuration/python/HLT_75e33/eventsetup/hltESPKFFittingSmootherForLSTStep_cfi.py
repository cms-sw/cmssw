import FWCore.ParameterSet.Config as cms

# Final fit of the LST step: the default smoother rejects a track with more than 3 outlier hits, dropping only its
# outermost hits first, which kills displaced outer-tracker-only tracks that carry a few wrong inner hits.
hltESPKFFittingSmootherForLSTStep = cms.ESProducer("KFFittingSmootherESProducer",
    BreakTrajWith2ConsecutiveMissing = cms.bool(False),
    ComponentName = cms.string('hltESPKFFittingSmootherForLSTStep'),
    EstimateCut = cms.double(20.0),
    Fitter = cms.string('hltESPRKTrajectoryFitter'),
    HighEtaSwitch = cms.double(5.0),
    LogPixelProbabilityCut = cms.double(0.0),
    MaxFractionOutliers = cms.double(0.5),
    MaxNumberOfOutliers = cms.int32(6),
    MinDof = cms.int32(2),
    MinNumberOfHits = cms.int32(3),
    MinNumberOfHitsHighEta = cms.int32(5),
    NoInvalidHitsBeginEnd = cms.bool(True),
    NoOutliersBeginEnd = cms.bool(False),
    RejectTracks = cms.bool(True),
    Smoother = cms.string('hltESPRKTrajectorySmoother'),
    appendToDataLabel = cms.string('')
)
