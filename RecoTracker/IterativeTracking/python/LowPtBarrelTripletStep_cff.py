import FWCore.ParameterSet.Config as cms

# NEW CLUSTERS (remove previously used clusters)
lowPtBarrelTripletStepClusters = cms.EDProducer("TrackClusterRemover",
    clusterLessSolution= cms.bool(True),
    trajectories = cms.InputTag("lowPtForwardTripletStepTracks"),
    oldClusterRemovalInfo = cms.InputTag("lowPtForwardTripletStepClusters"),
    overrideTrkQuals = cms.InputTag('lowPtForwardTripletStepSelector','lowPtForwardTripletStep'),
    TrackQuality = cms.string('highPurity'),
    pixelClusters = cms.InputTag("siPixelClusters"),
    stripClusters = cms.InputTag("siStripClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(9.0)
    )
)

# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
lowPtBarrelTripletStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.clone()
lowPtBarrelTripletStepSeedLayers.BPix.skipClusters = cms.InputTag('lowPtBarrelTripletStepClusters')
lowPtBarrelTripletStepSeedLayers.FPix.skipClusters = cms.InputTag('lowPtBarrelTripletStepClusters')
lowPtBarrelTripletStepSeedLayers.layerList = cms.vstring('BPix1+BPix2+BPix3') 


# SEEDS
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import RegionPsetFomBeamSpotBlock
lowPtBarrelTripletStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone(
    RegionFactoryPSet = RegionPsetFomBeamSpotBlock.clone(
    ComponentName = cms.string('GlobalRegionProducerFromBeamSpot'),
    RegionPSet = RegionPsetFomBeamSpotBlock.RegionPSet.clone(
    ptMin = 0.2,
    originRadius = 0.03,
    nSigmaZ = 4.0
    )
    )
    )
lowPtBarrelTripletStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'lowPtBarrelTripletStepSeedLayers'

from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
lowPtBarrelTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet.ComponentName = 'LowPtClusterShapeSeedComparitor'


# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
lowPtBarrelTripletStepTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone(
    ComponentName = 'lowPtBarrelTripletStepTrajectoryFilter',
    filterPset = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.filterPset.clone(
    #maxLostHits = 3, # use LostHitFraction filter instead
    minimumNumberOfHits = 3,
    minPt = 0.1
    )
    )

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi
lowPtBarrelTripletStepChi2Est = TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi.Chi2MeasurementEstimator.clone(
    ComponentName = cms.string('lowPtBarrelTripletStepChi2Est'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(9.0) 
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
lowPtBarrelTripletStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone(
    ComponentName = 'lowPtBarrelTripletStepTrajectoryBuilder',
    MeasurementTrackerName = '',
    trajectoryFilterName = 'lowPtBarrelTripletStepTrajectoryFilter',
    clustersToSkip = cms.InputTag('lowPtBarrelTripletStepClusters'),
    maxCand = 3,

    #lostHitPenalty = cms.double(10.0), 

    estimator = cms.string('lowPtBarrelTripletStepChi2Est'),
    # 0.63 GeV is the maximum pT for a charged particle to loop within the 1.1m radius
    # of the outermost Tracker barrel layer (with B=3.8T)
    maxPtForLooperReconstruction = cms.double(0.63) 
    # set the variable to a negative value to turn-off the looper reconstruction 
    #maxPtForLooperReconstruction = cms.double(-1.) 
    )

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
lowPtBarrelTripletStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('lowPtBarrelTripletStepSeeds'),

    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),

    TrajectoryBuilder = 'lowPtBarrelTripletStepTrajectoryBuilder',
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True,
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterialForLoopers'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialForLoopersOpposite'),
        numberMeasurementsForFit = cms.int32(4)
    )
)

### Have to clone the KF fitters because only the PropagatorWithMaterialForLoopers (no RK) can be used for the
### reconstruction of loopers

# TRACK FITTING
import TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi
lowPtBarrelTripletStepKFTrajectoryFitter = TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi.KFTrajectoryFitter.clone(
    ComponentName = cms.string('lowPtBarrelTripletStepKFTrajectoryFitter'),
    Propagator = cms.string('PropagatorWithMaterialForLoopers')
)

import TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi
lowPtBarrelTripletStepKFTrajectorySmoother = TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi.KFTrajectorySmoother.clone(
    ComponentName = cms.string('lowPtBarrelTripletStepKFTrajectorySmoother'),
    Propagator = cms.string('PropagatorWithMaterialForLoopers'),
    errorRescaling = cms.double(10.0)
)

import TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi
lowPtBarrelTripletStepKFFittingSmoother = TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi.KFFittingSmoother.clone(
    ComponentName = cms.string('lowPtBarrelTripletStepKFFittingSmoother'),
    Fitter = cms.string('lowPtBarrelTripletStepKFTrajectoryFitter'),
    Smoother = cms.string('lowPtBarrelTripletStepKFTrajectorySmoother'),
    EstimateCut = cms.double(20.0),
    LogPixelProbabilityCut = cms.double(-14.0),                               
    MinNumberOfHits = cms.int32(3)
)



import RecoTracker.TrackProducer.TrackProducer_cfi
lowPtBarrelTripletStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'lowPtBarrelTripletStepTrackCandidates',
    AlgorithmName = cms.string('iter1'),
    Fitter = cms.string('lowPtBarrelTripletStepKFFittingSmoother'),
    #Propagator = cms.string('PropagatorWithMaterialForLoopers'),
    #NavigationSchool = cms.string('') ### Is the outerHitPattern filled correctly for loopers???
    )


# Final selection
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
lowPtBarrelTripletStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src='lowPtBarrelTripletStepTracks',
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
LowPtBarrelTripletStep = cms.Sequence(lowPtBarrelTripletStepClusters*
                                      lowPtBarrelTripletStepSeedLayers*
                                      lowPtBarrelTripletStepSeeds*
                                      lowPtBarrelTripletStepTrackCandidates*
                                      lowPtBarrelTripletStepTracks*
                                      lowPtBarrelTripletStepSelector)
