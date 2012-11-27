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

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi
muonSeededMeasurementEstimatorForInOut = TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi.Chi2MeasurementEstimator.clone(
    ComponentName = cms.string('muonSeededMeasurementEstimatorForInOut'),
    MaxChi2 = cms.double(400.0), ## was 30 ## TO BE TUNED
    nSigma  = cms.double(4.),    ## was 3  ## TO BE TUNED 
)
muonSeededMeasurementEstimatorForOutIn = TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi.Chi2MeasurementEstimator.clone(
    ComponentName = cms.string('muonSeededMeasurementEstimatorForOutIn'),
    MaxChi2 = cms.double(30.0), ## was 30 ## TO BE TUNED
    nSigma  = cms.double(3.),    ## was 3  ## TO BE TUNED 
)

###------------- TrajectoryFilter, defining selections on the trajectories while building them ----------------
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
muonSeededTrajectoryFilterForInOut = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone(
    ComponentName = cms.string('muonSeededTrajectoryFilterForInOut')
)
muonSeededTrajectoryFilterForInOut.filterPset.constantValueForLostHitsFractionFilter = 10 ## allow more lost hits
muonSeededTrajectoryFilterForInOut.filterPset.minimumNumberOfHits = 3 ## allow more lost hits

muonSeededTrajectoryFilterForOutIn = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone(
    ComponentName = cms.string('muonSeededTrajectoryFilterForOutIn')
)
muonSeededTrajectoryFilterForOutIn.filterPset.constantValueForLostHitsFractionFilter = 10 ## allow more lost hits
muonSeededTrajectoryFilterForOutIn.filterPset.minimumNumberOfHits = 5 ## allow more lost hits

###------------- TrajectoryBuilders ----------------
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
muonSeededTrajectoryBuilderForInOut = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone(
    ComponentName = cms.string('muonSeededTrajectoryBuilderForInOut'),
    foundHitBonus = cms.double(1000.0),  
    lostHitPenalty = cms.double(1.0),   
    maxCand   = cms.int32(5),
    estimator = cms.string('muonSeededMeasurementEstimatorForInOut'),
    trajectoryFilterName = cms.string('muonSeededTrajectoryFilterForInOut'),
    inOutTrajectoryFilterName = cms.string('muonSeededTrajectoryFilterForInOut'), # not sure if it is used
    minNrOfHitsForRebuild    = cms.int32(2),
    requireSeedHitsInRebuild = cms.bool(True), 
    keepOriginalIfRebuildFails = cms.bool(True), 
)
muonSeededTrajectoryBuilderForOutIn = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone(
    ComponentName = cms.string('muonSeededTrajectoryBuilderForOutIn'),
    foundHitBonus = cms.double(1000.0),  
    lostHitPenalty = cms.double(1.0),   
    maxCand   = cms.int32(3),
    estimator = cms.string('muonSeededMeasurementEstimatorForOutIn'),
    trajectoryFilterName = cms.string('muonSeededTrajectoryFilterForOutIn'),
    inOutTrajectoryFilterName = cms.string('muonSeededTrajectoryFilterForOutIn'), # not sure if it is used
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
    TrajectoryBuilder = cms.string("muonSeededTrajectoryBuilderForInOut"),
    TrajectoryCleaner = cms.string('muonSeededTrajectoryCleanerBySharedHits'),
    RedundantSeedCleaner = cms.string("none"), 
)
muonSeededTrackCandidatesOutIn = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag("muonSeededSeedsOutIn"),
    TrajectoryBuilder = cms.string("muonSeededTrajectoryBuilderForOutIn"),
    TrajectoryCleaner = cms.string('muonSeededTrajectoryCleanerBySharedHits'),
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(False),
)

######## TRACK PRODUCERS 
import RecoTracker.TrackProducer.TrackProducer_cfi
muonSeededTracksOutIn = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = cms.InputTag("muonSeededTrackCandidatesOutIn"),
    AlgorithmName = cms.string('iter10'),
    Fitter = cms.string("muonSeededFittingSmootherWithOutliersRejectionAndRK"),
)
muonSeededTracksInOut = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = cms.InputTag("muonSeededTrackCandidatesInOut"),
    AlgorithmName = cms.string('iter9'),
    Fitter = cms.string("muonSeededFittingSmootherWithOutliersRejectionAndRK"),
)

import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
muonSeededTracksInOutSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src='muonSeededTracksInOut',
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'muonSeededTracksInOutLoose',
            chi2n_par = 0.5,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 4,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 1.3, 4.0 ),
            dz_par1 = ( 1.3, 4.0 ),
            d0_par2 = ( 1.3, 4.0 ),
            dz_par2 = ( 1.3, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'muonSeededTracksInOutTight',
            preFilterName = 'muonSeededTracksInOutLoose',
            chi2n_par = 0.35,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 4,
            maxNumberLostLayers = 0,
            minNumber3DLayers = 3,
            d0_par1 = ( 1.1, 4.0 ),
            dz_par1 = ( 1.1, 4.0 ),
            d0_par2 = ( 1.1, 4.0 ),
            dz_par2 = ( 1.1, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'muonSeededTracksInOutHighPurity',
            preFilterName = 'muonSeededTracksInOutTight',
            chi2n_par = 0.2,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 4,
            maxNumberLostLayers = 0,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.9, 4.0 ),
            dz_par1 = ( 0.9, 4.0 ),
            d0_par2 = ( 0.9, 4.0 ),
            dz_par2 = ( 0.9, 4.0 )
            ),
        ) #end of vpset
    ) #end of clone

muonSeededTracksOutInSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src='muonSeededTracksOutIn',
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'muonSeededTracksOutInLoose',
            chi2n_par = 0.5,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 4,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 1.3, 4.0 ),
            dz_par1 = ( 1.3, 4.0 ),
            d0_par2 = ( 1.3, 4.0 ),
            dz_par2 = ( 1.3, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'muonSeededTracksOutInTight',
            preFilterName = 'muonSeededTracksOutInLoose',
            chi2n_par = 0.35,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 4,
            maxNumberLostLayers = 0,
            minNumber3DLayers = 3,
            d0_par1 = ( 1.1, 4.0 ),
            dz_par1 = ( 1.1, 4.0 ),
            d0_par2 = ( 1.1, 4.0 ),
            dz_par2 = ( 1.1, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'muonSeededTracksOutInHighPurity',
            preFilterName = 'muonSeededTracksOutInTight',
            chi2n_par = 0.2,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 4,
            maxNumberLostLayers = 0,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.9, 4.0 ),
            dz_par1 = ( 0.9, 4.0 ),
            d0_par2 = ( 0.9, 4.0 ),
            dz_par2 = ( 0.9, 4.0 )
            ),
        ) #end of vpset
    ) #end of clone




muonSeededStepCore = cms.Sequence(
    muonSeededSeedsInOut + muonSeededTrackCandidatesInOut + muonSeededTracksInOut +
    muonSeededSeedsOutIn + muonSeededTrackCandidatesOutIn + muonSeededTracksOutIn 
)
muonSeededStepExtra = cms.Sequence(
    muonSeededTracksInOutSelector +
    muonSeededTracksOutInSelector
)

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
muonSeededStepDebug = cms.Sequence(
    muonSeededSeedsOutInAsTracks + muonSeededTrackCandidatesOutInAsTracks +
    muonSeededSeedsInOutAsTracks + muonSeededTrackCandidatesInOutAsTracks
)
