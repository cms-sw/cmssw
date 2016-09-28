import FWCore.ParameterSet.Config as cms

###### Muon reconstruction module #####
from RecoMuon.MuonIdentification.earlyMuons_cfi import earlyMuons

###### SEEDER MODELS ######
import RecoTracker.SpecialSeedGenerators.outInSeedsFromStandaloneMuons_cfi
import RecoTracker.SpecialSeedGenerators.inOutSeedsFromTrackerMuons_cfi
muonSeededSeedsOutIn = RecoTracker.SpecialSeedGenerators.outInSeedsFromStandaloneMuons_cfi.outInSeedsFromStandaloneMuons.clone(
    src = "earlyMuons",
)
muonSeededSeedsInOut = RecoTracker.SpecialSeedGenerators.inOutSeedsFromTrackerMuons_cfi.inOutSeedsFromTrackerMuons.clone(
    src = "earlyMuons",
)
### This is also needed for seeding
from RecoTracker.SpecialSeedGenerators.outInSeedsFromStandaloneMuons_cfi import hitCollectorForOutInMuonSeeds

###### EVENT-SETUP STUFF #######
###---------- Trajectory Cleaner, deciding how overlapping track candidates are arbitrated  ----------------
import TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi 
muonSeededTrajectoryCleanerBySharedHits = TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi.trajectoryCleanerBySharedHits.clone(
    ComponentName = cms.string('muonSeededTrajectoryCleanerBySharedHits'),
    fractionShared = cms.double(0.1),
    ValidHitBonus = cms.double(1000.0),
    MissingHitPenalty = cms.double(1.0),
    ComponentType = cms.string('TrajectoryCleanerBySharedHits'),
    allowSharedFirstHit = cms.bool(True)
)

###------------- MeasurementEstimator, defining the searcgh window for pattern recongnition ----------------

from TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi import Chi2MeasurementEstimator as _Chi2MeasurementEstimator
_muonSeededMeasurementEstimatorForInOutBase = _Chi2MeasurementEstimator.clone(
    ComponentName = cms.string('muonSeededMeasurementEstimatorForInOut'),
    MaxChi2 = cms.double(80.0), ## was 30 ## TO BE TUNED
    nSigma  = cms.double(4.),    ## was 3  ## TO BE TUNED 
)
muonSeededMeasurementEstimatorForInOut = _muonSeededMeasurementEstimatorForInOutBase.clone(
    MaxSagitta = cms.double(-1.)
)
from Configuration.Eras.Modifier_trackingPhase1PU70_cff import trackingPhase1PU70
trackingPhase1PU70.toReplaceWith(muonSeededMeasurementEstimatorForInOut, _muonSeededMeasurementEstimatorForInOutBase.clone(
    MaxChi2 = 400
))
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toModify(muonSeededMeasurementEstimatorForInOut, MaxChi2 = 400.0, MaxSagitta = 2)

_muonSeededMeasurementEstimatorForOutInBase = _Chi2MeasurementEstimator.clone(
    ComponentName = cms.string('muonSeededMeasurementEstimatorForOutIn'),
    MaxChi2 = cms.double(30.0), ## was 30 ## TO BE TUNED
    nSigma  = cms.double(3.),    ## was 3  ## TO BE TUNED
)
muonSeededMeasurementEstimatorForOutIn = _muonSeededMeasurementEstimatorForOutInBase.clone(
    MaxSagitta = cms.double(-1.) 
)
trackingPhase1PU70.toReplaceWith(muonSeededMeasurementEstimatorForOutIn, _muonSeededMeasurementEstimatorForOutInBase)

###------------- TrajectoryFilter, defining selections on the trajectories while building them ----------------
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
muonSeededTrajectoryFilterForInOut = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone()
muonSeededTrajectoryFilterForInOut.constantValueForLostHitsFractionFilter = 10 ## allow more lost hits
muonSeededTrajectoryFilterForInOut.minimumNumberOfHits = 3 ## allow more lost hits

muonSeededTrajectoryFilterForOutIn = muonSeededTrajectoryFilterForInOut.clone()
muonSeededTrajectoryFilterForOutIn.constantValueForLostHitsFractionFilter = 10 ## allow more lost hits
muonSeededTrajectoryFilterForOutIn.minimumNumberOfHits = 5 ## allow more lost hits

###------------- TrajectoryBuilders ----------------
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
muonSeededTrajectoryBuilderForInOut = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    foundHitBonus = cms.double(1000.0),  
    lostHitPenalty = cms.double(1.0),   
    maxCand   = cms.int32(5),
    estimator = cms.string('muonSeededMeasurementEstimatorForInOut'),
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('muonSeededTrajectoryFilterForInOut')),
    inOutTrajectoryFilter = cms.PSet(refToPSet_ = cms.string('muonSeededTrajectoryFilterForInOut')), # not sure if it is used
    minNrOfHitsForRebuild    = cms.int32(2),
    requireSeedHitsInRebuild = cms.bool(True), 
    keepOriginalIfRebuildFails = cms.bool(True), 
)
muonSeededTrajectoryBuilderForOutIn = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    foundHitBonus = cms.double(1000.0),  
    lostHitPenalty = cms.double(1.0),   
    maxCand   = cms.int32(3),
    estimator = cms.string('muonSeededMeasurementEstimatorForOutIn'),
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('muonSeededTrajectoryFilterForOutIn')),
    inOutTrajectoryFilter = cms.PSet(refToPSet_ = cms.string('muonSeededTrajectoryFilterForOutIn')), # not sure if it is used
    minNrOfHitsForRebuild    = cms.int32(5),
    requireSeedHitsInRebuild = cms.bool(True), 
    keepOriginalIfRebuildFails = cms.bool(False), 
)

###-------------  Fitter-Smoother -------------------
import TrackingTools.TrackFitters.RungeKuttaFitters_cff
muonSeededFittingSmootherWithOutliersRejectionAndRK = TrackingTools.TrackFitters.RungeKuttaFitters_cff.KFFittingSmootherWithOutliersRejectionAndRK.clone(
    ComponentName = cms.string("muonSeededFittingSmootherWithOutliersRejectionAndRK"),
    BreakTrajWith2ConsecutiveMissing = cms.bool(False), 
    EstimateCut = cms.double(50.), ## was 20.
)

######## TRACK CANDIDATE MAKERS
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
muonSeededTrackCandidatesInOut = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag("muonSeededSeedsInOut"),
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string("muonSeededTrajectoryBuilderForInOut")),
    TrajectoryCleaner = cms.string('muonSeededTrajectoryCleanerBySharedHits'),
    RedundantSeedCleaner = cms.string("none"), 
)
muonSeededTrackCandidatesOutIn = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag("muonSeededSeedsOutIn"),
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string("muonSeededTrajectoryBuilderForOutIn")),
    TrajectoryCleaner = cms.string('muonSeededTrajectoryCleanerBySharedHits'),
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(False),
)

######## TRACK PRODUCERS 
import RecoTracker.TrackProducer.TrackProducer_cfi
muonSeededTracksOutIn = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = cms.InputTag("muonSeededTrackCandidatesOutIn"),
    AlgorithmName = cms.string('muonSeededStepOutIn'),
    Fitter = cms.string("muonSeededFittingSmootherWithOutliersRejectionAndRK"),
)
muonSeededTracksInOut = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = cms.InputTag("muonSeededTrackCandidatesInOut"),
    AlgorithmName = cms.string('muonSeededStepInOut'),
    Fitter = cms.string("muonSeededFittingSmootherWithOutliersRejectionAndRK"),
)


# Final Classifier
from RecoTracker.FinalTrackSelectors.TrackCutClassifier_cff import *
muonSeededTracksInOutClassifier = TrackCutClassifier.clone()
muonSeededTracksInOutClassifier.src='muonSeededTracksInOut'
muonSeededTracksInOutClassifier.mva.minPixelHits = [0,0,0]
muonSeededTracksInOutClassifier.mva.maxChi2 = [9999.,9999.,9999.]
muonSeededTracksInOutClassifier.mva.maxChi2n = [10.0,1.0,0.4]
muonSeededTracksInOutClassifier.mva.minLayers = [3,5,5]
muonSeededTracksInOutClassifier.mva.min3DLayers = [1,2,2]
muonSeededTracksInOutClassifier.mva.maxLostLayers = [4,3,2]


muonSeededTracksOutInClassifier = TrackCutClassifier.clone()
muonSeededTracksOutInClassifier.src='muonSeededTracksOutIn'
muonSeededTracksOutInClassifier.mva.minPixelHits = [0,0,0]
muonSeededTracksOutInClassifier.mva.maxChi2 = [9999.,9999.,9999.]
muonSeededTracksOutInClassifier.mva.maxChi2n = [10.0,1.0,0.4]
muonSeededTracksOutInClassifier.mva.minLayers = [3,5,5]
muonSeededTracksOutInClassifier.mva.min3DLayers = [1,2,2]
muonSeededTracksOutInClassifier.mva.maxLostLayers = [4,3,2]


# For Phase1PU70
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
muonSeededTracksInOutSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src='muonSeededTracksInOut',
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'muonSeededTracksInOutLoose',
            applyAdaptedPVCuts = cms.bool(False),
            chi2n_par = 10.0,
            minNumberLayers = 3,
            min_nhits = 5,
            maxNumberLostLayers = 4,
            minNumber3DLayers = 0,
            minHitsToBypassChecks = 7
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'muonSeededTracksInOutTight',
            preFilterName = 'muonSeededTracksInOutLoose',
            applyAdaptedPVCuts = cms.bool(False),
            chi2n_par = 1.0,
            minNumberLayers = 5,
            min_nhits = 6,
            maxNumberLostLayers = 3,
            minNumber3DLayers = 2,
            minHitsToBypassChecks = 10
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'muonSeededTracksInOutHighPurity',
            preFilterName = 'muonSeededTracksInOutTight',
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
muonSeededTracksOutInSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src='muonSeededTracksOutIn',
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'muonSeededTracksOutInLoose',
            applyAdaptedPVCuts = cms.bool(False),
            chi2n_par = 10.0,
            minNumberLayers = 3,
            min_nhits = 5,
            maxNumberLostLayers = 4,
            minNumber3DLayers = 0,
            minHitsToBypassChecks = 7
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'muonSeededTracksOutInTight',
            preFilterName = 'muonSeededTracksOutInLoose',
            applyAdaptedPVCuts = cms.bool(False),
            chi2n_par = 1.0,
            minNumberLayers = 5,
            min_nhits = 6,
            maxNumberLostLayers = 3,
            minNumber3DLayers = 2,
            minHitsToBypassChecks = 10
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'muonSeededTracksOutInHighPurity',
            preFilterName = 'muonSeededTracksOutInTight',
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



muonSeededStepCoreInOut = cms.Sequence(
    muonSeededSeedsInOut + muonSeededTrackCandidatesInOut + muonSeededTracksInOut
)
muonSeededStepCore = cms.Sequence(
    muonSeededStepCoreInOut +
    muonSeededSeedsOutIn + muonSeededTrackCandidatesOutIn + muonSeededTracksOutIn
)
#Phase2 : just muon Seed InOut is used in this moment
trackingPhase2PU140.toReplaceWith(muonSeededStepCore, muonSeededStepCoreInOut)
muonSeededStepExtraInOut = cms.Sequence(
    muonSeededTracksInOutClassifier
)
trackingPhase1PU70.toReplaceWith(muonSeededStepExtraInOut, cms.Sequence(
    muonSeededTracksInOutSelector
))
trackingPhase2PU140.toReplaceWith(muonSeededStepExtraInOut, cms.Sequence(
    muonSeededTracksInOutSelector
))
muonSeededStepExtra = cms.Sequence(
    muonSeededStepExtraInOut +
    muonSeededTracksOutInClassifier
)
trackingPhase1PU70.toReplaceWith(muonSeededStepExtra, cms.Sequence(
    muonSeededStepExtraInOut +
    muonSeededTracksOutInSelector
))
trackingPhase2PU140.toReplaceWith(muonSeededStepExtra, cms.Sequence(
    muonSeededStepExtraInOut
))

muonSeededStep = cms.Sequence(
    earlyMuons +
    muonSeededStepCore +
    muonSeededStepExtra 
)
    
    
##### MODULES FOR DEBUGGING ###############3
muonSeededSeedsInOutAsTracks = cms.EDProducer("FakeTrackProducerFromSeed", src = cms.InputTag("muonSeededSeedsInOut"))
muonSeededSeedsOutInAsTracks = cms.EDProducer("FakeTrackProducerFromSeed", src = cms.InputTag("muonSeededSeedsOutIn"))
muonSeededTrackCandidatesInOutAsTracks = cms.EDProducer("FakeTrackProducerFromCandidate", src = cms.InputTag("muonSeededTrackCandidatesInOut"))
muonSeededTrackCandidatesOutInAsTracks = cms.EDProducer("FakeTrackProducerFromCandidate", src = cms.InputTag("muonSeededTrackCandidatesOutIn"))
muonSeededStepDebugInOut = cms.Sequence(
    muonSeededSeedsInOutAsTracks + muonSeededTrackCandidatesInOutAsTracks
)
muonSeededStepDebug = cms.Sequence(
    muonSeededSeedsOutInAsTracks + muonSeededTrackCandidatesOutInAsTracks +
    muonSeededStepDebugInOut
)
