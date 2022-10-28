import FWCore.ParameterSet.Config as cms

###### Muon reconstruction module #####
from RecoMuon.MuonIdentification.earlyMuons_cfi import earlyMuons

###### SEEDER MODELS ######
import RecoTracker.SpecialSeedGenerators.outInSeedsFromStandaloneMuons_cfi
import RecoTracker.SpecialSeedGenerators.inOutSeedsFromTrackerMuons_cfi
muonSeededSeedsOutIn = RecoTracker.SpecialSeedGenerators.outInSeedsFromStandaloneMuons_cfi.outInSeedsFromStandaloneMuons.clone(
    src = 'earlyMuons',
)
muonSeededSeedsInOut = RecoTracker.SpecialSeedGenerators.inOutSeedsFromTrackerMuons_cfi.inOutSeedsFromTrackerMuons.clone(
    src = 'earlyMuons',
)
### This is also needed for seeding
from RecoTracker.SpecialSeedGenerators.outInSeedsFromStandaloneMuons_cfi import hitCollectorForOutInMuonSeeds
from Configuration.Eras.Modifier_tracker_apv_vfp30_2016_cff import tracker_apv_vfp30_2016 as _tracker_apv_vfp30_2016
_tracker_apv_vfp30_2016.toModify(hitCollectorForOutInMuonSeeds, MinPtForHitRecoveryInGluedDet=1e9)

###### EVENT-SETUP STUFF #######
###---------- Trajectory Cleaner, deciding how overlapping track candidates are arbitrated  ----------------
import TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi 
muonSeededTrajectoryCleanerBySharedHits = TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi.trajectoryCleanerBySharedHits.clone(
    ComponentName       = 'muonSeededTrajectoryCleanerBySharedHits',
    fractionShared      = 0.1,
    ValidHitBonus       = 1000.0,
    MissingHitPenalty   = 1.0,
    ComponentType       = 'TrajectoryCleanerBySharedHits',
    allowSharedFirstHit = True
)

###------------- MeasurementEstimator, defining the searcgh window for pattern recongnition ----------------

from TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi import Chi2MeasurementEstimator as _Chi2MeasurementEstimator
_muonSeededMeasurementEstimatorForInOutBase = _Chi2MeasurementEstimator.clone(
    ComponentName = 'muonSeededMeasurementEstimatorForInOut',
    MaxChi2       = 80.0, ## was 30 ## TO BE TUNED
    nSigma        = 4.,    ## was 3  ## TO BE TUNED 
)
muonSeededMeasurementEstimatorForInOut = _muonSeededMeasurementEstimatorForInOutBase.clone(
    MaxSagitta = -1.
)
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toModify(muonSeededMeasurementEstimatorForInOut, MaxChi2 = 400.0, MaxSagitta = 2)

_muonSeededMeasurementEstimatorForOutInBase = _Chi2MeasurementEstimator.clone(
    ComponentName = 'muonSeededMeasurementEstimatorForOutIn',
    MaxChi2       = 30.0, ## was 30 ## TO BE TUNED
    nSigma        = 3.,    ## was 3  ## TO BE TUNED
)
from Configuration.Eras.Modifier_tracker_apv_vfp30_2016_cff import tracker_apv_vfp30_2016 as _tracker_apv_vfp30_2016
_tracker_apv_vfp30_2016.toModify(_muonSeededMeasurementEstimatorForOutInBase, MinPtForHitRecoveryInGluedDet=1e9)
muonSeededMeasurementEstimatorForOutIn = _muonSeededMeasurementEstimatorForOutInBase.clone(
    MaxSagitta = -1. 
)

###------------- TrajectoryFilter, defining selections on the trajectories while building them ----------------
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
muonSeededTrajectoryFilterForInOut = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    constantValueForLostHitsFractionFilter = 10, ## allow more lost hits
    minimumNumberOfHits                    = 3 ## allow more lost hits
)
muonSeededTrajectoryFilterForOutIn = muonSeededTrajectoryFilterForInOut.clone(
    constantValueForLostHitsFractionFilter = 10, ## allow more lost hits
    minimumNumberOfHits = 5 ## allow more lost hits
)
###------------- TrajectoryBuilders ----------------
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
muonSeededTrajectoryBuilderForInOut = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    foundHitBonus    = 1000.0,
    lostHitPenalty   = 1.0,
    maxCand          = 5,
    estimator        = 'muonSeededMeasurementEstimatorForInOut',
    trajectoryFilter = dict(refToPSet_ = 'muonSeededTrajectoryFilterForInOut'),
    inOutTrajectoryFilter      = dict(refToPSet_ = 'muonSeededTrajectoryFilterForInOut'), # not sure if it is used
    minNrOfHitsForRebuild      = 2,
    requireSeedHitsInRebuild   = True, 
    keepOriginalIfRebuildFails = True, 
)
muonSeededTrajectoryBuilderForOutIn = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    foundHitBonus    = 1000.0,
    lostHitPenalty   = 1.0,
    maxCand          = 3,
    estimator        = 'muonSeededMeasurementEstimatorForOutIn',
    trajectoryFilter = dict(refToPSet_ = 'muonSeededTrajectoryFilterForOutIn'),
    inOutTrajectoryFilter      = dict(refToPSet_ = 'muonSeededTrajectoryFilterForOutIn'), # not sure if it is used
    minNrOfHitsForRebuild      = 5,
    requireSeedHitsInRebuild   = True,
    keepOriginalIfRebuildFails = False,
)

###-------------  Fitter-Smoother -------------------
import TrackingTools.TrackFitters.RungeKuttaFitters_cff
muonSeededFittingSmootherWithOutliersRejectionAndRK = TrackingTools.TrackFitters.RungeKuttaFitters_cff.KFFittingSmootherWithOutliersRejectionAndRK.clone(
    ComponentName = 'muonSeededFittingSmootherWithOutliersRejectionAndRK',
    BreakTrajWith2ConsecutiveMissing = False, 
    EstimateCut   = 50., ## was 20.
)

######## TRACK CANDIDATE MAKERS
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
muonSeededTrackCandidatesInOut = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = 'muonSeededSeedsInOut',
    TrajectoryBuilderPSet = dict(refToPSet_ = 'muonSeededTrajectoryBuilderForInOut'),
    TrajectoryCleaner     = 'muonSeededTrajectoryCleanerBySharedHits',
    RedundantSeedCleaner  = 'none', 
)
muonSeededTrackCandidatesOutIn = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = 'muonSeededSeedsOutIn',
    TrajectoryBuilderPSet       = dict(refToPSet_ = 'muonSeededTrajectoryBuilderForOutIn'),
    TrajectoryCleaner           = 'muonSeededTrajectoryCleanerBySharedHits',
    numHitsForSeedCleaner       = 50,
    onlyPixelHitsForSeedCleaner = False,
)

######## TRACK PRODUCERS 
import RecoTracker.TrackProducer.TrackProducer_cfi
muonSeededTracksOutIn = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src           = 'muonSeededTrackCandidatesOutIn',
    AlgorithmName = 'muonSeededStepOutIn',
    Fitter        = 'muonSeededFittingSmootherWithOutliersRejectionAndRK',
)
muonSeededTracksInOut = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src           = 'muonSeededTrackCandidatesInOut',
    AlgorithmName = 'muonSeededStepInOut',
    Fitter        = 'muonSeededFittingSmootherWithOutliersRejectionAndRK',
)

# Final Classifier
from RecoTracker.FinalTrackSelectors.TrackCutClassifier_cff import *
muonSeededTracksInOutClassifier = TrackCutClassifier.clone(
    src = 'muonSeededTracksInOut',
    mva = dict(
	minPixelHits  = [0,0,0],
        maxChi2       = [9999.,9999.,9999.],
        maxChi2n      = [10.0,1.0,0.4],
        minLayers     = [3,5,5],
        min3DLayers   = [1,2,2],
        maxLostLayers = [4,3,2]
    )
)

muonSeededTracksOutInClassifier = TrackCutClassifier.clone(
    src = 'muonSeededTracksOutIn',
    mva = dict(
	minPixelHits  = [0,0,0],
        maxChi2       = [9999.,9999.,9999.],
        maxChi2n      = [10.0,1.0,0.4],
        minLayers     = [3,5,5],
        min3DLayers   = [1,2,2],
        maxLostLayers = [4,3,2]
    )
)

from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toModify(muonSeededTracksOutInClassifier.mva,
                  dr_par = cms.PSet(
                      d0err = cms.vdouble(0.003, 0.003, 0.003),
                      d0err_par = cms.vdouble(0.001, 0.001, 0.001),
                      dr_exp = cms.vint32(4, 4, 4),
                      dr_par1 = cms.vdouble(0.4, 0.4, 0.4),
                      dr_par2 = cms.vdouble(0.3, 0.3, 0.3)
                  ),
                  dz_par = cms.PSet(
                      dz_exp = cms.vint32(4, 4, 4),
                      dz_par1 = cms.vdouble(0.4, 0.4, 0.4),
                      dz_par2 = cms.vdouble(0.35, 0.35, 0.35)
                  ),
                  maxDr         = [9999.,9999.,0.5],
                  maxDz         = [9999.,9999.,0.5],
                  minHits     =   [0,0,10],
                  minPixelHits  = [0,0,1],
)
pp_on_AA.toModify(muonSeededTracksInOutClassifier.mva,
                  dr_par = cms.PSet(
                      d0err = cms.vdouble(0.003, 0.003, 0.003),
                      d0err_par = cms.vdouble(0.001, 0.001, 0.001),
                      dr_exp = cms.vint32(4, 4, 4),
                      dr_par1 = cms.vdouble(0.4, 0.4, 0.4),
                      dr_par2 = cms.vdouble(0.3, 0.3, 0.3)
                  ),
                  dz_par = cms.PSet(
                      dz_exp = cms.vint32(4, 4, 4),
                      dz_par1 = cms.vdouble(0.4, 0.4, 0.4),
                      dz_par2 = cms.vdouble(0.35, 0.35, 0.35)
                  ),
                  maxDr         = [9999.,9999.,0.5],
                  maxDz         = [9999.,9999.,0.5],
                  minHits     =   [0,0,10],
                  minPixelHits  = [0,0,1],
)

# For Phase2PU140
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
muonSeededTracksInOutSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src='muonSeededTracksInOut',
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name                  = 'muonSeededTracksInOutLoose',
            applyAdaptedPVCuts    = False,
            chi2n_par             = 10.0,
            minNumberLayers       = 3,
            min_nhits             = 5,
            maxNumberLostLayers   = 4,
            minNumber3DLayers     = 0,
            minHitsToBypassChecks = 7
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name                  = 'muonSeededTracksInOutTight',
            preFilterName         = 'muonSeededTracksInOutLoose',
            applyAdaptedPVCuts    = False,
            chi2n_par             = 1.0,
            minNumberLayers       = 5,
            min_nhits             = 6,
            maxNumberLostLayers   = 3,
            minNumber3DLayers     = 2,
            minHitsToBypassChecks = 10
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name                  = 'muonSeededTracksInOutHighPurity',
            preFilterName         = 'muonSeededTracksInOutTight',
            applyAdaptedPVCuts    = False,
            chi2n_par             = 0.4,
            minNumberLayers       = 5,
            min_nhits             = 7,
            maxNumberLostLayers   = 2,
            minNumber3DLayers     = 2,
            minHitsToBypassChecks = 20
            ),
        ) #end of vpset
    ) #end of clone
muonSeededTracksOutInSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src='muonSeededTracksOutIn',
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name                  = 'muonSeededTracksOutInLoose',
            applyAdaptedPVCuts    = False,
            chi2n_par             = 10.0,
            minNumberLayers       = 3,
            min_nhits             = 5,
            maxNumberLostLayers   = 4,
            minNumber3DLayers     = 0,
            minHitsToBypassChecks = 7
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name                  = 'muonSeededTracksOutInTight',
            preFilterName         = 'muonSeededTracksOutInLoose',
            applyAdaptedPVCuts    = False,
            chi2n_par             = 1.0,
            minNumberLayers       = 5,
            min_nhits             = 6,
            maxNumberLostLayers   = 3,
            minNumber3DLayers     = 2,
            minHitsToBypassChecks = 10
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name                  = 'muonSeededTracksOutInHighPurity',
            preFilterName         = 'muonSeededTracksOutInTight',
            applyAdaptedPVCuts    = False,
            chi2n_par             = 0.4,
            minNumberLayers       = 5,
            min_nhits             = 7,
            maxNumberLostLayers   = 2,
            minNumber3DLayers     = 2,
            minHitsToBypassChecks = 20
            ),
        ) #end of vpset
    ) #end of clone



muonSeededStepCoreInOutTask = cms.Task(
    muonSeededSeedsInOut , muonSeededTrackCandidatesInOut , muonSeededTracksInOut
)
muonSeededStepCoreInOut = cms.Sequence(muonSeededStepCoreInOutTask)

muonSeededStepCoreOutInTask = cms.Task(
    muonSeededSeedsOutIn , muonSeededTrackCandidatesOutIn , muonSeededTracksOutIn
)
muonSeededStepCoreOutIn = cms.Sequence(muonSeededStepCoreOutInTask)

muonSeededStepCoreTask = cms.Task(
    muonSeededStepCoreInOutTask ,
    muonSeededStepCoreOutInTask
)
muonSeededStepCore = cms.Sequence(muonSeededStepCoreTask)
#Phase2 : just muon Seed InOut is used in this moment
#trackingPhase2PU140.toReplaceWith(muonSeededStepCore, muonSeededStepCoreInOut)

from Configuration.Eras.Modifier_phase2_timing_layer_cff import phase2_timing_layer
phase2_timing_layer.toModify(muonSeededTracksInOut, TrajectoryInEvent = True)
phase2_timing_layer.toModify(muonSeededTracksOutIn, TrajectoryInEvent = True)

muonSeededStepExtraInOutTask = cms.Task(
    muonSeededTracksInOutClassifier
)
muonSeededStepExtraInOut = cms.Sequence(muonSeededStepExtraInOutTask)

trackingPhase2PU140.toReplaceWith(muonSeededStepExtraInOutTask, cms.Task(
    muonSeededTracksInOutSelector
))

muonSeededStepExtraTask = cms.Task(
    muonSeededStepExtraInOutTask ,
    muonSeededTracksOutInClassifier
)

muonSeededStepExtra = cms.Sequence(muonSeededStepExtraTask)
trackingPhase2PU140.toReplaceWith(muonSeededStepExtraTask, cms.Task(
    muonSeededStepExtraInOutTask ,
    muonSeededTracksOutInSelector
))

muonSeededStepTask = cms.Task(
    earlyMuons,
    muonSeededStepCoreTask,
    muonSeededStepExtraTask 
)
muonSeededStep = cms.Sequence(muonSeededStepTask) 
   
    
##### MODULES FOR DEBUGGING ###############3
muonSeededSeedsInOutAsTracks = cms.EDProducer('FakeTrackProducerFromSeed', src = cms.InputTag('muonSeededSeedsInOut'))
muonSeededSeedsOutInAsTracks = cms.EDProducer('FakeTrackProducerFromSeed', src = cms.InputTag('muonSeededSeedsOutIn'))
muonSeededTrackCandidatesInOutAsTracks = cms.EDProducer('FakeTrackProducerFromCandidate', src = cms.InputTag('muonSeededTrackCandidatesInOut'))
muonSeededTrackCandidatesOutInAsTracks = cms.EDProducer('FakeTrackProducerFromCandidate', src = cms.InputTag('muonSeededTrackCandidatesOutIn'))
muonSeededStepDebugInOutTask = cms.Task(
    muonSeededSeedsInOutAsTracks , muonSeededTrackCandidatesInOutAsTracks
)
muonSeededStepDebugInOut = cms.Sequence(muonSeededStepDebugInOutTask)
muonSeededStepDebugTask = cms.Task(
    muonSeededSeedsOutInAsTracks , muonSeededTrackCandidatesOutInAsTracks ,
    muonSeededStepDebugInOutTask
)
muonSeededStepDebug = cms.Sequence(muonSeededStepDebugTask)
