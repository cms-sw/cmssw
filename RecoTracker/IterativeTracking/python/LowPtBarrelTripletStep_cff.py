import FWCore.ParameterSet.Config as cms

# for no-loopers
from Configuration.ProcessModifiers.trackingNoLoopers_cff import trackingNoLoopers

# NEW CLUSTERS (remove previously used clusters)
from RecoLocalTracker.SubCollectionProducers.trackClusterRemover_cfi import *
lowPtBarrelTripletStepClusters = trackClusterRemover.clone(
    maxChi2               = 9.0,
    trajectories          = 'lowPtForwardTripletStepTracks',
    pixelClusters         = 'siPixelClusters',
    stripClusters         = 'siStripClusters',
    oldClusterRemovalInfo = 'lowPtForwardTripletStepClusters',
    overrideTrkQuals      = 'lowPtForwardTripletStepSelector:lowPtForwardTripletStep',
    TrackQuality          = 'highPurity'
)

# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
lowPtBarrelTripletStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.clone(
    BPix      = dict(skipClusters = cms.InputTag('lowPtBarrelTripletStepClusters')),
    FPix      = dict(skipClusters = cms.InputTag('lowPtBarrelTripletStepClusters')),
    layerList = ['BPix1+BPix2+BPix3'] 
)

# SEEDS
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import RegionPsetFomBeamSpotBlock
lowPtBarrelTripletStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone(
    RegionFactoryPSet = RegionPsetFomBeamSpotBlock.clone(
    ComponentName     = 'GlobalRegionProducerFromBeamSpot',
    RegionPSet        = RegionPsetFomBeamSpotBlock.RegionPSet.clone(
    ptMin             = 0.2,
    originRadius      = 0.03,
    nSigmaZ           = 4.0 )
    )
)
lowPtBarrelTripletStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'lowPtBarrelTripletStepSeedLayers'

from RecoTracker.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
import RecoTracker.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
lowPtBarrelTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet = RecoTracker.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor


# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
lowPtBarrelTripletStepTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    #maxLostHits = 3, # use LostHitFraction filter instead
    minimumNumberOfHits = 3,
    minPt               = 0.1
)

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi
lowPtBarrelTripletStepChi2Est = TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi.Chi2MeasurementEstimator.clone(
    ComponentName = 'lowPtBarrelTripletStepChi2Est',
    nSigma        = 3.0,
    MaxChi2       = 9.0 
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
lowPtBarrelTripletStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    trajectoryFilter = dict(refToPSet_ = 'lowPtBarrelTripletStepTrajectoryFilter'),
    clustersToSkip = 'lowPtBarrelTripletStepClusters',
    maxCand = 3,
    #lostHitPenalty = 10.,
    estimator = 'lowPtBarrelTripletStepChi2Est',
    # 0.63 GeV is the maximum pT for a charged particle to loop within the 1.1m radius
    # of the outermost Tracker barrel layer (with B=3.8T)
    maxPtForLooperReconstruction = 0.63,
    # set the variable to a negative value to turn-off the looper reconstruction
    #maxPtForLooperReconstruction = -1.,
)
trackingNoLoopers.toModify(lowPtBarrelTripletStepTrajectoryBuilder,
                           maxPtForLooperReconstruction = 0.0)
# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
lowPtBarrelTripletStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = 'lowPtBarrelTripletStepSeeds',
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = 50,
    onlyPixelHitsForSeedCleaner = True,
    TrajectoryBuilderPSet = dict(refToPSet_ = 'lowPtBarrelTripletStepTrajectoryBuilder'),
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True,
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = 'PropagatorWithMaterialForLoopers',
        propagatorOppositeTISE = 'PropagatorWithMaterialForLoopersOpposite',
        numberMeasurementsForFit = 4,
    )
)

### Have to clone the KF fitters because only the PropagatorWithMaterialForLoopers (no RK) can be used for the
### reconstruction of loopers

# TRACK FITTING
import TrackingTools.TrackFitters.KFTrajectoryFitter_cfi
lowPtBarrelTripletStepKFTrajectoryFitter = TrackingTools.TrackFitters.KFTrajectoryFitter_cfi.KFTrajectoryFitter.clone(
    ComponentName = 'lowPtBarrelTripletStepKFTrajectoryFitter',
    Propagator    = 'PropagatorWithMaterialForLoopers'
)

import TrackingTools.TrackFitters.KFTrajectorySmoother_cfi
lowPtBarrelTripletStepKFTrajectorySmoother = TrackingTools.TrackFitters.KFTrajectorySmoother_cfi.KFTrajectorySmoother.clone(
    ComponentName  = 'lowPtBarrelTripletStepKFTrajectorySmoother',
    Propagator     = 'PropagatorWithMaterialForLoopers',
    errorRescaling = 10.0
)

import TrackingTools.TrackFitters.KFFittingSmoother_cfi
lowPtBarrelTripletStepKFFittingSmoother = TrackingTools.TrackFitters.KFFittingSmoother_cfi.KFFittingSmoother.clone(
    ComponentName          = 'lowPtBarrelTripletStepKFFittingSmoother',
    Fitter                 = 'lowPtBarrelTripletStepKFTrajectoryFitter',
    Smoother               = 'lowPtBarrelTripletStepKFTrajectorySmoother',
    EstimateCut            = 20.0,
    LogPixelProbabilityCut = -14.0,                               
    MinNumberOfHits        = 3
)

import RecoTracker.TrackProducer.TrackProducerIterativeDefault_cfi
lowPtBarrelTripletStepTracks = RecoTracker.TrackProducer.TrackProducerIterativeDefault_cfi.TrackProducer.clone(
    src           = 'lowPtBarrelTripletStepTrackCandidates',
    AlgorithmName = 'lowPtTripletStep',
    Fitter        = 'lowPtBarrelTripletStepKFFittingSmoother',
    #Propagator = cms.string('PropagatorWithMaterialForLoopers'),
    #NavigationSchool = cms.string('') ### Is the outerHitPattern filled correctly for loopers???
)


# Final selection
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
lowPtBarrelTripletStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src = 'lowPtBarrelTripletStepTracks',
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'lowPtBarrelTripletStepLoose',
            ), #end of pset
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'lowPtBarrelTripletStepTight',
            preFilterName = 'lowPtBarrelTripletStepLoose',
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'lowPtBarrelTripletStep',
            preFilterName = 'lowPtBarrelTripletStepTight',
            ),
    ) #end of vpset
) #end of clone

# Final sequence
LowPtBarrelTripletStepTask = cms.Task(lowPtBarrelTripletStepClusters,
                                      lowPtBarrelTripletStepSeedLayers,
                                      lowPtBarrelTripletStepSeeds,
                                      lowPtBarrelTripletStepTrackCandidates,
                                      lowPtBarrelTripletStepTracks,
                                      lowPtBarrelTripletStepSelector)
LowPtBarrelTripletStep = cms.Sequence(LowPtBarrelTripletStep)
