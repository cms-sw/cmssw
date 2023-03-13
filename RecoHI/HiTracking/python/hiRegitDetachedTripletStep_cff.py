import FWCore.ParameterSet.Config as cms

################################################################################### 
# pp iterative tracking modified for hiOffline reco (the vertex is the one reconstructed in HI)
################################### 3rd step: low-pT and displaced tracks from pixel triplets

from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *

###################################
import RecoTracker.IterativeTracking.DetachedTripletStep_cff

# NEW CLUSTERS (remove previously used clusters)
hiRegitDetachedTripletStepClusters = cms.EDProducer("HITrackClusterRemover",
                                                clusterLessSolution= cms.bool(True),
                                                oldClusterRemovalInfo = cms.InputTag("hiRegitPixelPairStepClusters"),
                                                trajectories = cms.InputTag("hiRegitPixelPairStepTracks"),
                                                overrideTrkQuals = cms.InputTag('hiRegitPixelPairStepSelector','hiRegitPixelPairStep'),
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



# SEEDING LAYERS
hiRegitDetachedTripletStepSeedLayers =  RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepSeedLayers.clone(
    BPix = dict(skipClusters = 'hiRegitDetachedTripletStepClusters'),
    FPix = dict(skipClusters = 'hiRegitDetachedTripletStepClusters')
)

from RecoTracker.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
#import RecoTracker.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
# seeding
hiRegitDetachedTripletStepSeeds     = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepSeeds.clone(
    RegionFactoryPSet = HiTrackingRegionFactoryFromJetsBlock.clone(
    	RegionPSet = dict(ptMin = 1.2)
    ),
    ClusterCheckPSet = dict(doClusterCheck = False), # do not check for max number of clusters pixel or strips
    OrderedHitsFactoryPSet = dict(
	SeedingLayers = 'hiRegitDetachedTripletStepSeedLayers'
       #GeneratorPSet = dict(SeedComparitorPSet = RecoTracker.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor.clone()),
    ),
)
# building: feed the new-named seeds
hiRegitDetachedTripletStepTrajectoryFilter = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepTrajectoryFilterBase.clone()
hiRegitDetachedTripletStepTrajectoryBuilder = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepTrajectoryBuilder.clone(
    trajectoryFilter     = dict(refToPSet_ = 'hiRegitDetachedTripletStepTrajectoryFilter'),
    clustersToSkip       = 'hiRegitDetachedTripletStepClusters'
)

hiRegitDetachedTripletStepTrackCandidates        =  RecoTracker.IterativeTracking.DetachedTripletStep_cff._detachedTripletStepTrackCandidatesCkf.clone(
    src               = 'hiRegitDetachedTripletStepSeeds',
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('hiRegitDetachedTripletStepTrajectoryBuilder')),
    maxNSeeds=100000
)

# fitting: feed new-names
hiRegitDetachedTripletStepTracks                 = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepTracks.clone(
    src                 = 'hiRegitDetachedTripletStepTrackCandidates',
    #AlgorithmName = 'jetCoreRegionalStep',
    AlgorithmName = 'detachedTripletStep',
)


# Track selection
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiRegitDetachedTripletStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src = 'hiRegitDetachedTripletStepTracks',
    trackSelectors = cms.VPSet(
       RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
           name = 'hiRegitDetachedTripletStepLoose',
           d0_par2 = [9999.0, 0.0],
           dz_par2 = [9999.0, 0.0],
           applyAdaptedPVCuts = False
       ), #end of pset
       RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
           name = 'hiRegitDetachedTripletStepTight',
           preFilterName = 'hiRegitDetachedTripletStepLoose',
           d0_par2 = [9999.0, 0.0],
           dz_par2 = [9999.0, 0.0],
           applyAdaptedPVCuts = False
       ),
       RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
           name = 'hiRegitDetachedTripletStep',
           preFilterName = 'hiRegitDetachedTripletStepTight',
           d0_par2 = [9999.0, 0.0],
           dz_par2 = [9999.0, 0.0],
       applyAdaptedPVCuts = False
    ),
  ) #end of vpset
) #end of clone  


hiRegitDetachedTripletStepTask = cms.Task(hiRegitDetachedTripletStepClusters,
                                          hiRegitDetachedTripletStepSeedLayers,
                                          hiRegitDetachedTripletStepSeeds,
                                          hiRegitDetachedTripletStepTrackCandidates,
                                          hiRegitDetachedTripletStepTracks,
                                          hiRegitDetachedTripletStepSelector
                                          )
hiRegitDetachedTripletStep = cms.Sequence(hiRegitDetachedTripletStepTask)
