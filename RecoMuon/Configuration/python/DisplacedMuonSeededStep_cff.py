import FWCore.ParameterSet.Config as cms

###### Muon reconstruction module #####
from RecoMuon.MuonIdentification.earlyMuons_cfi import earlyDisplacedMuons
from RecoTracker.IterativeTracking.MuonSeededStep_cff import *

###### SEEDER MODELS ######
#for displaced global muons
muonSeededSeedsOutInDisplaced = RecoTracker.SpecialSeedGenerators.outInSeedsFromStandaloneMuons_cfi.outInSeedsFromStandaloneMuons.clone(
    src = "earlyDisplacedMuons",
)
muonSeededSeedsOutInDisplaced.fromVertex = cms.bool(False)
###------------- MeasurementEstimator, defining the searcgh window for pattern recongnition ----------------
#for displaced global muons
muonSeededMeasurementEstimatorForOutInDisplaced = TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi.Chi2MeasurementEstimator.clone(
    ComponentName = cms.string('muonSeededMeasurementEstimatorForOutInDisplaced'),
    MaxChi2 = cms.double(30.0), ## was 30 ## TO BE TUNED
    nSigma  = cms.double(3.),    ## was 3  ## TO BE TUNED 
)

###------------- TrajectoryFilter, defining selections on the trajectories while building them ----------------
#for displaced global muons
muonSeededTrajectoryFilterForOutInDisplaced = muonSeededTrajectoryFilterForInOut.clone()
muonSeededTrajectoryFilterForOutInDisplaced.constantValueForLostHitsFractionFilter = 10 ## allow more lost hits
muonSeededTrajectoryFilterForOutInDisplaced.minimumNumberOfHits = 5 ## allow more lost hits
###------------- TrajectoryBuilders ----------------
#for displaced global muons
muonSeededTrajectoryBuilderForOutInDisplaced = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    foundHitBonus = cms.double(1000.0),  
    lostHitPenalty = cms.double(1.0),   
    maxCand   = cms.int32(3),
    estimator = cms.string('muonSeededMeasurementEstimatorForOutInDisplaced'),
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('muonSeededTrajectoryFilterForOutInDisplaced')),
    inOutTrajectoryFilter = cms.PSet(refToPSet_ = cms.string('muonSeededTrajectoryFilterForOutInDisplaced')), # not sure if it is used
    minNrOfHitsForRebuild    = cms.int32(5),
    requireSeedHitsInRebuild = cms.bool(True), 
    keepOriginalIfRebuildFails = cms.bool(False), 
)
######## TRACK CANDIDATE MAKERS
#for displaced global muons
muonSeededTrackCandidatesOutInDisplaced = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag("muonSeededSeedsOutInDisplaced"),
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string("muonSeededTrajectoryBuilderForOutInDisplaced")),
    TrajectoryCleaner = cms.string('muonSeededTrajectoryCleanerBySharedHits'),
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(False),
)

######## TRACK PRODUCERS 
#for displaced global muon
muonSeededTracksOutInDisplaced = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = cms.InputTag("muonSeededTrackCandidatesOutInDisplaced"),
    AlgorithmName = cms.string('muonSeededStepOutIn'),
    Fitter = cms.string("muonSeededFittingSmootherWithOutliersRejectionAndRK"),
)

#for displaced global muons
muonSeededTracksOutInDisplacedSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src='muonSeededTracksOutInDisplaced',
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'muonSeededTracksOutInDisplacedLoose',
            applyAdaptedPVCuts = cms.bool(False),
            chi2n_par = 10.0,
            minNumberLayers = 3,
            min_nhits = 5,
            maxNumberLostLayers = 4,
            minNumber3DLayers = 0,
            minHitsToBypassChecks = 7
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'muonSeededTracksOutInDisplacedTight',
            preFilterName = 'muonSeededTracksOutInDisplacedLoose',
            applyAdaptedPVCuts = cms.bool(False),
            chi2n_par = 1.0,
            minNumberLayers = 5,
            min_nhits = 6,
            maxNumberLostLayers = 3,
            minNumber3DLayers = 2,
            minHitsToBypassChecks = 10
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'muonSeededTracksOutInDisplacedHighPurity',
            preFilterName = 'muonSeededTracksOutInDisplacedTight',
            applyAdaptedPVCuts = cms.bool(False),
            chi2n_par = 0.4,
            minNumberLayers = 5,
            min_nhits = 7,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 2,
            minHitsToBypassChecks = 20
            ),
        ) #end of vpset
    ) #end of clone

#for displaced global muons
muonSeededStepCoreDisplaced = cms.Sequence(
    muonSeededSeedsInOut + muonSeededTrackCandidatesInOut + muonSeededTracksInOut +
    muonSeededSeedsOutInDisplaced + muonSeededTrackCandidatesOutInDisplaced + muonSeededTracksOutInDisplaced 
)

#for displaced global muons
muonSeededStepExtraDisplaced = cms.Sequence(
    muonSeededTracksInOutSelector +
    muonSeededTracksOutInDisplacedSelector
)
#for displaced global muons
muonSeededStepDisplaced = cms.Sequence(
    earlyDisplacedMuons +
    muonSeededStepCoreDisplaced +
    muonSeededStepExtraDisplaced 
)
    
##### MODULES FOR DEBUGGING ###############3
#for displaced global muons
muonSeededSeedsOutInDisplacedAsTracks = cms.EDProducer("FakeTrackProducerFromSeed", src = cms.InputTag("muonSeededSeedsOutInDisplaced"))
#for displaced global muons
muonSeededTrackCandidatesOutInDisplacedAsTracks = cms.EDProducer("FakeTrackProducerFromCandidate", src = cms.InputTag("muonSeededTrackCandidatesOutInDisplaced"))
#for displaced global muons
muonSeededStepDebugDisplaced = cms.Sequence(
    muonSeededSeedsOutInDisplacedAsTracks + muonSeededTrackCandidatesOutInDisplacedAsTracks +
    muonSeededSeedsInOutAsTracks + muonSeededTrackCandidatesInOutAsTracks
)
