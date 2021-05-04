import FWCore.ParameterSet.Config as cms

################################################################################### 
# pp iterative tracking modified for hiOffline reco (the vertex is the one reconstructed in HI)
################################### 4th step: large impact parameter tracking using mixed-triplet seeding

from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *

###################################
from RecoTracker.IterativeTracking.MixedTripletStep_cff import *

# NEW CLUSTERS (remove previously used clusters)
hiRegitMixedTripletStepClusters = cms.EDProducer("HITrackClusterRemover",
                                                clusterLessSolution= cms.bool(True),
                                                oldClusterRemovalInfo = cms.InputTag("hiRegitDetachedTripletStepClusters"),
                                                trajectories = cms.InputTag("hiRegitDetachedTripletStepTracks"),
                                                overrideTrkQuals = cms.InputTag('hiRegitDetachedTripletStepSelector','hiRegitDetachedTripletStep'),
                                                TrackQuality = cms.string('highPurity'),
                                                pixelClusters = cms.InputTag("siPixelClusters"),
                                                stripClusters = cms.InputTag("siStripClusters"),
                                                Common = cms.PSet(
   							 maxChi2 = cms.double(9.0),
   							 ),
   					        Strip = cms.PSet(
   							 maxChi2 = cms.double(9.0),
   							 #Yen-Jie's mod to preserve merged clusters
   							 maxSize = cms.uint32(2)
   							 )
                                                )



# SEEDING LAYERS A
hiRegitMixedTripletStepSeedLayersA =  RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedLayersA.clone(
    BPix = dict(skipClusters = 'hiRegitMixedTripletStepClusters'),
    FPix = dict(skipClusters = 'hiRegitMixedTripletStepClusters'),
    TEC  = dict(skipClusters = 'hiRegitMixedTripletStepClusters'),
    layerList = ['BPix1+BPix2+BPix3',
                 'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg',
                 'BPix1+FPix1_pos+FPix2_pos', 'BPix1+FPix1_neg+FPix2_neg',
                 'BPix2+FPix1_pos+FPix2_pos', 'BPix2+FPix1_neg+FPix2_neg',
                 'FPix1_pos+FPix2_pos+TEC1_pos', 'FPix1_neg+FPix2_neg+TEC1_neg']
)
# SEEDS A
hiRegitMixedTripletStepSeedsA = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedsA.clone(
    RegionFactoryPSet = HiTrackingRegionFactoryFromJetsBlock.clone(
	RegionPSet = dict(ptMin = 1.0)
    ),
    ClusterCheckPSet = dict(doClusterCheck = False), # do not check for max number of clusters pixel or strips
    OrderedHitsFactoryPSet = dict(SeedingLayers = 'hiRegitMixedTripletStepSeedLayersA'),
)
# SEEDING LAYERS B
hiRegitMixedTripletStepSeedLayersB =  RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedLayersB.clone(
    BPix = dict(skipClusters = 'hiRegitMixedTripletStepClusters'),
    TIB  = dict(skipClusters = 'hiRegitMixedTripletStepClusters'),
    layerList = ['BPix2+BPix3+TIB1','BPix2+BPix3+TIB2']
)
hiRegitMixedTripletStepSeedsB = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedsB.clone(
    RegionFactoryPSet = HiTrackingRegionFactoryFromJetsBlock.clone(
	RegionPSet = dict(ptMin = 1.0)
    ),
    ClusterCheckPSet = dict(doClusterCheck = False), # do not check for max number of clusters pixel or strips
    OrderedHitsFactoryPSet = dict(SeedingLayers = 'hiRegitMixedTripletStepSeedLayersB'),
)
# combine seeds
hiRegitMixedTripletStepSeeds = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeeds.clone(
    seedCollections = ['hiRegitMixedTripletStepSeedsA',
                       'hiRegitMixedTripletStepSeedsB'
                      ],
)

# track building
hiRegitMixedTripletStepTrajectoryFilter = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepTrajectoryFilter.clone()

hiRegitMixedTripletStepTrajectoryBuilder = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepTrajectoryBuilder.clone(
    trajectoryFilter     = cms.PSet(refToPSet_ = cms.string('hiRegitMixedTripletStepTrajectoryFilter')),
    clustersToSkip       = cms.InputTag('hiRegitMixedTripletStepClusters'),
)

hiRegitMixedTripletStepTrackCandidates        =  RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepTrackCandidates.clone(
    src               = 'hiRegitMixedTripletStepSeeds',
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('hiRegitMixedTripletStepTrajectoryBuilder')),
    maxNSeeds = 100000
    )

# fitting: feed new-names
hiRegitMixedTripletStepTracks                 = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepTracks.clone(
    src                 = 'hiRegitMixedTripletStepTrackCandidates',
    #AlgorithmName = 'conversionStep',
    AlgorithmName = 'mixedTripletStep',
)

# Track selection
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiRegitMixedTripletStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src = 'hiRegitMixedTripletStepTracks',
    trackSelectors = cms.VPSet(
       RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
           name = 'hiRegitMixedTripletStepLoose',
           d0_par2 = [9999.0, 0.0],
           dz_par2 = [9999.0, 0.0],
           applyAdaptedPVCuts = False
       ), #end of pset
       RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
           name = 'hiRegitMixedTripletStepTight',
           preFilterName = 'hiRegitMixedTripletStepLoose',
           d0_par2 = [9999.0, 0.0],
           dz_par2 = [9999.0, 0.0],
           applyAdaptedPVCuts = False
       ),
       RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
           name = 'hiRegitMixedTripletStep',
           preFilterName = 'hiRegitMixedTripletStepTight',
           d0_par2 = [9999.0, 0.0],
           dz_par2 = [9999.0, 0.0],
           applyAdaptedPVCuts = False
       ),
    ) #end of vpset
) #end of clone  

hiRegitMixedTripletStepTask = cms.Task(hiRegitMixedTripletStepClusters,
                                       hiRegitMixedTripletStepSeedLayersA,
                                       hiRegitMixedTripletStepSeedsA,
                                       hiRegitMixedTripletStepSeedLayersB,
                                       hiRegitMixedTripletStepSeedsB,
                                       hiRegitMixedTripletStepSeeds,
                                       hiRegitMixedTripletStepTrackCandidates,
                                       hiRegitMixedTripletStepTracks,
                                       hiRegitMixedTripletStepSelector
                                       )
hiRegitMixedTripletStep = cms.Sequence(hiRegitMixedTripletStepTask)
