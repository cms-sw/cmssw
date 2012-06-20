####################################################################
# Does two tracking iterations and then merges them.               #
# Step 0 uses pixel-triplet seeding.                               #
# Step 1 uses mixed-pair seeding.                                  #
####################################################################

### STEP 0 ###

# hit building
from RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff import *
from RecoTracker.TransientTrackingRecHit.TTRHBuilders_cff import *

# These 3 are not used anywhere. Can they be removed ?
from RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cff import *
from RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff import *
from RecoTracker.TkSeedGenerator.GlobalPixelSeeds_cff import *

# seeding
from RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff import *
from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import RegionPsetFomBeamSpotBlock
newSeedFromTriplets = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone(
    RegionFactoryPSet = RegionPsetFomBeamSpotBlock.clone(
    ComponentName = cms.string('GlobalRegionProducerFromBeamSpot'),
    RegionPSet = RegionPsetFomBeamSpotBlock.RegionPSet.clone(
    ptMin = 0.8
    )
    )
    )
newSeedFromTriplets.ClusterCheckPSet.PixelClusterCollectionLabel = 'siPixelClusters'
newSeedFromTriplets.ClusterCheckPSet.ClusterCollectionLabel = 'siStripClusters'
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
newSeedFromTriplets.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet.ComponentName = 'LowPtClusterShapeSeedComparitor'

# building
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
newTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone(
    ComponentName = 'newTrajectoryFilter',
    filterPset = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.filterPset.clone(
    minimumNumberOfHits = 3,
    minPt = 0.6
    )
    )

import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
newTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone(
    ComponentName = 'newTrajectoryBuilder',
    trajectoryFilterName = 'newTrajectoryFilter'
    )

from RecoTracker.CkfPattern.CkfTrackCandidates_cff import *
newTrackCandidateMaker = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('newSeedFromTriplets'),
    TrajectoryBuilder = 'newTrajectoryBuilder'
    )

# fitting
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff import *
preFilterZeroStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'newTrackCandidateMaker',
    AlgorithmName = 'iter0'
    )

### STEP 1 ###


# new hit collection
newClusters = cms.EDProducer("TrackClusterRemover",
    trajectories = cms.InputTag("preFilterZeroStepTracks"),
    overrideTrkQuals = cms.InputTag('zeroSelector','zeroStepTracksWithQuality'),                         
    TrackQuality = cms.string('highPurity'),                         
    pixelClusters = cms.InputTag("siPixelClusters"),
    stripClusters = cms.InputTag("siStripClusters"),
    clusterLessSolution= cms.bool(True),
    Common = cms.PSet(
        maxChi2 = cms.double(30.0)
    )
)

# make corresponding rechits
#import RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi
#import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi
#newPixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone(
#    src = cms.InputTag("newClusters")
#    )
#newStripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone(
#    ClusterProducer = 'newClusters'
#    )


# seeding 
newMixedLayerPairs = RecoTracker.TkSeedingLayers.MixedLayerPairs_cfi.mixedlayerpairs.clone(
    ComponentName = 'newMixedLayerPairs'
    )
newMixedLayerPairs.BPix.HitProducer = 'siPixelRecHits'
newMixedLayerPairs.BPix.skipClusters = cms.InputTag('newClusters')
newMixedLayerPairs.FPix.HitProducer = 'siPixelRecHits'
newMixedLayerPairs.FPix.skipClusters = cms.InputTag('newClusters')
#newMixedLayerPairs.TEC.matchedRecHits = cms.InputTag("newStripRecHits","matchedRecHit")
newMixedLayerPairs.TEC.matchedRecHits = cms.InputTag('siStripMatchedRecHits','matchedRecHit')
newMixedLayerPairs.TEC.skipClusters = cms.InputTag('newClusters')

from RecoTracker.TkSeedGenerator.GlobalSeedsFromPairsWithVertices_cff import *
newSeedFromPairs = RecoTracker.TkSeedGenerator.GlobalSeedsFromPairsWithVertices_cff.globalSeedsFromPairsWithVertices.clone()
newSeedFromPairs.RegionFactoryPSet.RegionPSet.ptMin = 0.6
newSeedFromPairs.RegionFactoryPSet.RegionPSet.originRadius = 0.05
newSeedFromPairs.OrderedHitsFactoryPSet.SeedingLayers = cms.string('newMixedLayerPairs')
newSeedFromPairs.ClusterCheckPSet.PixelClusterCollectionLabel = 'siPixelClusters'
newSeedFromPairs.ClusterCheckPSet.ClusterCollectionLabel = 'siStripClusters'
   

# building 
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
newMeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone(
    ComponentName = 'newMeasurementTracker',
    skipClusters = cms.InputTag('newClusters'),
    pixelClusterProducer = 'siPixelClusters',
    stripClusterProducer = 'siStripClusters'
    )

import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
stepOneTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone(
    ComponentName = 'stepOneTrajectoryFilter',
    filterPset = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.filterPset.clone(
    minimumNumberOfHits = 3,
    minPt = 0.4
    )
    )

stepOneCkfTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone(
    ComponentName = 'stepOneCkfTrajectoryBuilder',
    MeasurementTrackerName = '',
    trajectoryFilterName = 'stepOneTrajectoryFilter',
    clustersToSkip = cms.InputTag('newClusters')
    )

stepOneTrackCandidateMaker = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('newSeedFromPairs'),
    TrajectoryBuilder = 'stepOneCkfTrajectoryBuilder'
    )


# fitting
preFilterStepOneTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    AlgorithmName = 'iter1',
    src = 'stepOneTrackCandidateMaker',
    )

### BOTH STEPS TOGETHER ###

# Set track quality flags for both steps

import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
zeroSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src='preFilterZeroStepTracks',
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'zeroStepWithLooseQuality',
            ), #end of pset
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'zeroStepWithTightQuality',
            preFilterName = 'zeroStepWithLooseQuality',
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'zeroStepTracksWithQuality',
            preFilterName = 'zeroStepWithTightQuality',
            ),
        ) #end of vpset
    ) #end of clone

firstSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src='preFilterStepOneTracks',
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'firstStepWithLooseQuality',
            ), #end of pset
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'firstStepWithTightQuality',
            preFilterName = 'firstStepWithLooseQuality',
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'preMergingFirstStepTracksWithQuality',
            preFilterName = 'firstStepWithTightQuality',
            ),
        ) #end of vpset
    ) #end of clone



import RecoTracker.FinalTrackSelectors.trackListMerger_cfi

#then merge everything together
firstStepTracksWithQuality = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = cms.VInputTag(cms.InputTag('preFilterZeroStepTracks'),cms.InputTag('preFilterStepOneTracks')),
    hasSelector=cms.vint32(1,1),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("zeroSelector","zeroStepTracksWithQuality"),cms.InputTag("firstSelector","preMergingFirstStepTracksWithQuality")),
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1), pQual=cms.bool(False) ))
)                        

# Final sequence
firstStep = cms.Sequence(newSeedFromTriplets*newTrackCandidateMaker*preFilterZeroStepTracks*zeroSelector*
                         newClusters*
                         newSeedFromPairs*stepOneTrackCandidateMaker*preFilterStepOneTracks*
                         firstSelector*firstStepTracksWithQuality)






