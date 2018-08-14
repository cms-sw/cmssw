import FWCore.ParameterSet.Config as cms

###### Muon reconstruction module #####
from RecoMuon.MuonIdentification.earlyMuons_cfi import earlyDisplacedMuons

###### SEEDER MODELS ######
#for displaced global muons
import RecoTracker.SpecialSeedGenerators.outInSeedsFromStandaloneMuons_cfi
muonSeededSeedsOutInDisplaced = RecoTracker.SpecialSeedGenerators.outInSeedsFromStandaloneMuons_cfi.outInSeedsFromStandaloneMuons.clone(
    src = "earlyDisplacedMuons",
)
muonSeededSeedsOutInDisplaced.fromVertex = cms.bool(False)
###------------- MeasurementEstimator, defining the searcgh window for pattern recongnition ----------------
#for displaced global muons
import TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi
muonSeededMeasurementEstimatorForOutInDisplaced = TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi.Chi2MeasurementEstimator.clone(
    ComponentName = cms.string('muonSeededMeasurementEstimatorForOutInDisplaced'),
    MaxChi2 = cms.double(30.0), ## was 30 ## TO BE TUNED
    nSigma  = cms.double(3.),    ## was 3  ## TO BE TUNED 
)

###------------- TrajectoryFilter, defining selections on the trajectories while building them ----------------
#for displaced global muons
import RecoTracker.IterativeTracking.MuonSeededStep_cff
muonSeededTrajectoryFilterForOutInDisplaced = RecoTracker.IterativeTracking.MuonSeededStep_cff.muonSeededTrajectoryFilterForInOut.clone()
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
muonSeededTracksOutInDisplacedClassifier = RecoTracker.IterativeTracking.MuonSeededStep_cff.muonSeededTracksOutInClassifier.clone()
muonSeededTracksOutInDisplacedClassifier.src='muonSeededTracksOutInDisplaced'


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
