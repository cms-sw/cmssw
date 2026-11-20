import FWCore.ParameterSet.Config as cms

hltPhase2MuonIOTrackCandidates = cms.EDProducer("CkfTrackCandidateMaker",
    MeasurementTrackerEvent = cms.InputTag("hltMeasurementTrackerEvent"),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    RedundantSeedCleaner = cms.string('none'),
    TrajectoryBuilderPSet = cms.PSet(
        refToPSet_ = cms.string('HLTIter0Phase2L3FromL1TkMuonPSetGroupedCkfTrajectoryBuilderIT')
    ),
    TrajectoryCleaner = cms.string('hltESPTrajectoryCleanerBySharedHits'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        numberMeasurementsForFit = cms.int32(4),
        propagatorAlongTISE = cms.string('PropagatorWithMaterialParabolicMf'),
        propagatorOppositeTISE = cms.string('hltOppositeMaterialPropagatorParabolicMF')
    ),
    cleanTrajectoryAfterInOut = cms.bool(False),
    doSeedingRegionRebuilding = cms.bool(True),
    maxNSeeds = cms.uint32(100000),
    maxSeedsBeforeCleaning = cms.uint32(1000),
    src = cms.InputTag("hltPhase2MuonPixelSeedsFromPixelTracks"),
    useHitsSplitting = cms.bool(True)
)

_hltPhase2MuonIOTrackCandidates = cms.EDProducer('MkFitOutputConverter',
    batchSize = cms.int32(16),
    candMVASel = cms.bool(False),
    candCutSel = cms.bool(True),
    candMinNHitsCut = cms.int32(4),
    candMinPtCut = cms.double(0.9),
    candMinPtRelaxedCut = cms.double(0.8),
    candMinAbsEtaForRelaxedCut = cms.double(1.4),
    candWP = cms.double(0),
    doErrorRescale = cms.bool(True),
    mightGet = cms.optional.untracked.vstring,
    mkFitEventOfHits = cms.InputTag("hltMkFitEventOfHits"),
    mkFitPixelHits = cms.InputTag("hltMkFitSiPixelHits"),
    mkFitSeeds = cms.InputTag("hltPhase2MuonIOTrackMkFitSeeds"),
    mkFitStripHits = cms.InputTag("hltMkFitSiPhase2Hits"),
    propagatorAlong = cms.ESInputTag("","PropagatorWithMaterial"),
    propagatorOpposite = cms.ESInputTag("","PropagatorWithMaterialOpposite"),
    qualityMaxInvPt = cms.double(100),
    qualityMaxPosErr = cms.double(100),
    qualityMaxR = cms.double(120),
    qualityMaxZ = cms.double(280),
    qualityMinTheta = cms.double(0.01),
    qualitySignPt = cms.bool(True),
    seeds = cms.InputTag("hltPhase2MuonIOTrackSeeds"),
    tfDnnLabel = cms.string('trackSelectionTf'),
    tracks = cms.InputTag("hltPhase2MuonIOTrackCandidatesMkFit"),
    ttrhBuilder = cms.ESInputTag("","hltESPTTRHBuilderWithTrackAngle")
)

from Configuration.ProcessModifiers.phase2MuonSeedsSelector_cff import phase2MuonSeedsSelector
phase2MuonSeedsSelector.toReplaceWith(
    hltPhase2MuonIOTrackCandidates,
    _hltPhase2MuonIOTrackCandidates
)
