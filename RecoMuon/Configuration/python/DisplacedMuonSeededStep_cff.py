import FWCore.ParameterSet.Config as cms

###### Muon reconstruction module #####
from RecoMuon.MuonIdentification.earlyMuons_cfi import earlyDisplacedMuons

###### SEEDER MODELS ######
#for displaced global muons
import RecoTracker.SpecialSeedGenerators.outInSeedsFromStandaloneMuons_cfi
muonSeededSeedsOutInDisplaced = RecoTracker.SpecialSeedGenerators.outInSeedsFromStandaloneMuons_cfi.outInSeedsFromStandaloneMuons.clone(
    src        = "earlyDisplacedMuons",
    fromVertex = False
)
###------------- MeasurementEstimator, defining the searcgh window for pattern recongnition ----------------
#for displaced global muons
import TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi
muonSeededMeasurementEstimatorForOutInDisplaced = TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi.Chi2MeasurementEstimator.clone(
    ComponentName = 'muonSeededMeasurementEstimatorForOutInDisplaced',
    MaxChi2 = 30.0, ## was 30 ## TO BE TUNED
    nSigma  = 3.,    ## was 3  ## TO BE TUNED 
)

###------------- TrajectoryFilter, defining selections on the trajectories while building them ----------------
#for displaced global muons
import RecoTracker.IterativeTracking.MuonSeededStep_cff
muonSeededTrajectoryFilterForOutInDisplaced = RecoTracker.IterativeTracking.MuonSeededStep_cff.muonSeededTrajectoryFilterForInOut.clone(
    constantValueForLostHitsFractionFilter = 10, ## allow more lost hits
    minimumNumberOfHits = 5 ## allow more lost hits
)
###------------- TrajectoryBuilders ----------------
#for displaced global muons
muonSeededTrajectoryBuilderForOutInDisplaced = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    foundHitBonus = 1000.0,
    lostHitPenalty = 1.0,
    maxCand   = 3,
    estimator = 'muonSeededMeasurementEstimatorForOutInDisplaced',
    trajectoryFilter = dict(refToPSet_ = 'muonSeededTrajectoryFilterForOutInDisplaced'),
    inOutTrajectoryFilter = dict(refToPSet_ = 'muonSeededTrajectoryFilterForOutInDisplaced'), # not sure if it is used
    minNrOfHitsForRebuild    = 5,
    requireSeedHitsInRebuild = True, 
    keepOriginalIfRebuildFails = False, 
)
######## TRACK CANDIDATE MAKERS
#for displaced global muons
muonSeededTrackCandidatesOutInDisplaced = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = "muonSeededSeedsOutInDisplaced",
    TrajectoryBuilderPSet = dict(refToPSet_ = "muonSeededTrajectoryBuilderForOutInDisplaced"),
    TrajectoryCleaner = 'muonSeededTrajectoryCleanerBySharedHits',
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(False),
)

######## TRACK PRODUCERS 
#for displaced global muon
muonSeededTracksOutInDisplaced = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = "muonSeededTrackCandidatesOutInDisplaced",
    AlgorithmName = 'muonSeededStepOutIn',
    Fitter = "muonSeededFittingSmootherWithOutliersRejectionAndRK",
)

#for displaced global muons
muonSeededTracksOutInDisplacedClassifier = RecoTracker.IterativeTracking.MuonSeededStep_cff.muonSeededTracksOutInClassifier.clone(
    src='muonSeededTracksOutInDisplaced'
)

#for displaced global muons
muonSeededStepCoreDisplacedTask = cms.Task(
    cms.TaskPlaceholder("muonSeededStepCoreInOutTask"),
    muonSeededSeedsOutInDisplaced , muonSeededTrackCandidatesOutInDisplaced , muonSeededTracksOutInDisplaced
)
muonSeededStepCoreDisplaced = cms.Sequence(muonSeededStepCoreDisplacedTask)

#for displaced global muons
muonSeededStepExtraDisplacedTask = cms.Task(
    cms.TaskPlaceholder("muonSeededStepExtraInOutTask"),
    muonSeededTracksOutInDisplacedClassifier
)
muonSeededStepExtraDisplaced = cms.Sequence(muonSeededStepExtraDisplacedTask)

#for displaced global muons
muonSeededStepDisplacedTask = cms.Task(
    earlyDisplacedMuons ,
    muonSeededStepCoreDisplacedTask ,
    muonSeededStepExtraDisplacedTask 
)
muonSeededStepDisplaced = cms.Sequence(muonSeededStepDisplacedTask)
    
##### MODULES FOR DEBUGGING ###############3
#for displaced global muons
muonSeededSeedsOutInDisplacedAsTracks = cms.EDProducer("FakeTrackProducerFromSeed", src = cms.InputTag("muonSeededSeedsOutInDisplaced"))
#for displaced global muons
muonSeededTrackCandidatesOutInDisplacedAsTracks = cms.EDProducer("FakeTrackProducerFromCandidate", src = cms.InputTag("muonSeededTrackCandidatesOutInDisplaced"))
#for displaced global muons
muonSeededStepDebugDisplacedTask = cms.Task(
    cms.TaskPlaceholder("muonSeededStepDebugInOutTask"),
    muonSeededSeedsOutInDisplacedAsTracks , muonSeededTrackCandidatesOutInDisplacedAsTracks 
)
muonSeededStepDebugDisplaced = cms.Sequence(muonSeededStepDebugDisplacedTask)
