import FWCore.ParameterSet.Config as cms

################################################################################### 
# pp iterative tracking modified for hiOffline reco (the vertex is the one reconstructed in HI)
################################### 3rd step: low-pT and displaced tracks from pixel triplets

from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *

###################################
from RecoTracker.IterativeTracking.DetachedTripletStep_cff import *

# NEW CLUSTERS (remove previously used clusters)
hiRegitDetachedTripletStepClusters = cms.EDProducer("TrackClusterRemover",
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
hiRegitDetachedTripletStepSeedLayers =  RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepSeedLayers.clone()
hiRegitDetachedTripletStepSeedLayers.BPix.skipClusters = cms.InputTag('hiRegitDetachedTripletStepClusters')
hiRegitDetachedTripletStepSeedLayers.FPix.skipClusters = cms.InputTag('hiRegitDetachedTripletStepClusters')

# seeding
hiRegitDetachedTripletStepSeeds     = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepSeeds.clone()
hiRegitDetachedTripletStepSeeds.RegionFactoryPSet                                           = HiTrackingRegionFactoryFromJetsBlock.clone()
hiRegitDetachedTripletStepSeeds.ClusterCheckPSet.doClusterCheck                             = False # do not check for max number of clusters pixel or strips
hiRegitDetachedTripletStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'hiRegitDetachedTripletStepSeedLayers'
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
#hiRegitDetachedTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet.ComponentName = 'LowPtClusterShapeSeedComparitor'
hiRegitDetachedTripletStepSeeds.RegionFactoryPSet.RegionPSet.ptMin = 1.2

# building: feed the new-named seeds
hiRegitDetachedTripletStepTrajectoryFilter = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepTrajectoryFilter.clone(
    ComponentName    = 'hiRegitDetachedTripletStepTrajectoryFilter'
    )

hiRegitDetachedTripletStepTrajectoryBuilder = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepTrajectoryBuilder.clone(
    ComponentName        = 'hiRegitDetachedTripletStepTrajectoryBuilder',
    trajectoryFilterName = 'hiRegitDetachedTripletStepTrajectoryFilter',
    clustersToSkip       = cms.InputTag('hiRegitDetachedTripletStepClusters')
)

hiRegitDetachedTripletStepTrackCandidates        =  RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepTrackCandidates.clone(
    src               = cms.InputTag('hiRegitDetachedTripletStepSeeds'),
    TrajectoryBuilder = 'hiRegitDetachedTripletStepTrajectoryBuilder',
    maxNSeeds=100000
    )

# fitting: feed new-names
hiRegitDetachedTripletStepTracks                 = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepTracks.clone(
    src                 = 'hiRegitDetachedTripletStepTrackCandidates',
    #AlgorithmName = cms.string('iter7'),
    AlgorithmName = cms.string('iter3'),
    )


# Track selection
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiRegitDetachedTripletStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src='hiRegitDetachedTripletStepTracks',
    trackSelectors= cms.VPSet(
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


hiRegitDetachedTripletStep = cms.Sequence(hiRegitDetachedTripletStepClusters*
                                          hiRegitDetachedTripletStepSeedLayers*
                                          hiRegitDetachedTripletStepSeeds*
                                          hiRegitDetachedTripletStepTrackCandidates*
                                          hiRegitDetachedTripletStepTracks*
                                          hiRegitDetachedTripletStepSelector
                                          )

