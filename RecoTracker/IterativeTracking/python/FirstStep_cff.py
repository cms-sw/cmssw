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
from RecoTracker.TkSeedGenerator.GlobalSeedsFromTripletsWithVertices_cff import *
newSeedFromTriplets = RecoTracker.TkSeedGenerator.GlobalSeedsFromTripletsWithVertices_cff.globalSeedsFromTripletsWithVertices.clone()
newSeedFromTriplets.RegionFactoryPSet.RegionPSet.ptMin = 0.5

# building
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
newTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone()
newTrajectoryFilter.ComponentName = 'newTrajectoryFilter'
newTrajectoryFilter.filterPset.minimumNumberOfHits = 3
newTrajectoryFilter.filterPset.minPt = 0.3

import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
newTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone()
newTrajectoryBuilder.ComponentName = 'newTrajectoryBuilder'
newTrajectoryBuilder.trajectoryFilterName = 'newTrajectoryFilter'

from RecoTracker.CkfPattern.CkfTrackCandidates_cff import *
newTrackCandidateMaker = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
newTrackCandidateMaker.src = cms.InputTag('newSeedFromTriplets')
newTrackCandidateMaker.TrajectoryBuilder = 'newTrajectoryBuilder'

# fitting
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff import *
preFilterZeroStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone()
preFilterZeroStepTracks.src = 'newTrackCandidateMaker'
preFilterZeroStepTracks.Fitter = 'KFFittingSmootherWithOutliersRejectionAndRK'
preFilterZeroStepTracks.AlgorithmName = 'ctf'

### STEP 1 ###

# Lock hits from step0 tracks
zeroStepFilter = cms.EDFilter("QualityFilter",
    TrackQuality = cms.string('highPurity'),
    recTracks = cms.InputTag("zeroStepTracksWithQuality:")
)

# new hit collection
newClusters = cms.EDFilter("TrackClusterRemover",
    trajectories = cms.InputTag("zeroStepFilter"), 
    pixelClusters = cms.InputTag("siPixelClusters"),
    stripClusters = cms.InputTag("siStripClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(30.0)
    )
)

# make corresponding rechits
import RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi
import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi
newPixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone()
newStripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone()
newPixelRecHits.src = cms.InputTag("newClusters")
newStripRecHits.ClusterProducer = 'newClusters'


# seeding 
newMixedLayerPairs = RecoTracker.TkSeedingLayers.MixedLayerPairs_cfi.mixedlayerpairs.clone()
newMixedLayerPairs.ComponentName = 'newMixedLayerPairs'
newMixedLayerPairs.BPix.HitProducer = 'newPixelRecHits'
newMixedLayerPairs.FPix.HitProducer = 'newPixelRecHits'
newMixedLayerPairs.TEC.matchedRecHits = cms.InputTag("newStripRecHits","matchedRecHit")

from RecoTracker.TkSeedGenerator.GlobalSeedsFromPairsWithVertices_cff import *
newSeedFromPairs = RecoTracker.TkSeedGenerator.GlobalSeedsFromPairsWithVertices_cff.globalSeedsFromPairsWithVertices.clone()
newSeedFromPairs.RegionFactoryPSet.RegionPSet.ptMin = 0.9
newSeedFromPairs.OrderedHitsFactoryPSet.SeedingLayers = cms.string('newMixedLayerPairs')


# building 
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
newMeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone()
newMeasurementTracker.ComponentName = 'newMeasurementTracker'
newMeasurementTracker.pixelClusterProducer = 'newClusters'
newMeasurementTracker.stripClusterProducer = 'newClusters'

stepOneCkfTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone()
stepOneCkfTrajectoryBuilder.ComponentName = 'stepOneCkfTrajectoryBuilder'
stepOneCkfTrajectoryBuilder.MeasurementTrackerName = 'newMeasurementTracker'
stepOneCkfTrajectoryBuilder.trajectoryFilterName = 'newTrajectoryFilter'

stepOneTrackCandidateMaker = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
stepOneTrackCandidateMaker.src = cms.InputTag('newSeedFromPairs')
stepOneTrackCandidateMaker.TrajectoryBuilder = 'stepOneCkfTrajectoryBuilder'


# fitting
preFilterStepOneTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone()
preFilterStepOneTracks.AlgorithmName = cms.string('ctf')
preFilterStepOneTracks.src = 'stepOneTrackCandidateMaker'
preFilterStepOneTracks.clusterRemovalInfo = 'newClusters'

### BOTH STEPS TOGETHER ###

# Set track quality flags for both steps
from RecoTracker.FinalTrackSelectors.TracksWithQuality_cff import *

# Merge step 0 and step 1
from RecoTracker.FinalTrackSelectors.MergeTrackCollections_cff import *

# Final sequence
firstStep = cms.Sequence(newSeedFromTriplets*newTrackCandidateMaker*preFilterZeroStepTracks*tracksWithQualityZeroStep*
                         zeroStepFilter*newClusters*newPixelRecHits*newStripRecHits*
                         newSeedFromPairs*stepOneTrackCandidateMaker*preFilterStepOneTracks*tracksWithQualityStepOne*
                         firstStepTracksWithQuality)






