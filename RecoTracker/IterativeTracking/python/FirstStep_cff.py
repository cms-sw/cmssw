# hit building
from RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff import *
from RecoTracker.TransientTrackingRecHit.TTRHBuilders_cff import *

# seeding
from RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cff import *
from RecoTracker.TkSeedGenerator.GlobalSeedsFromPairsWithVertices_cff import *
from RecoTracker.TkSeedGenerator.GlobalSeedsFromTripletsWithVertices_cff import *
from RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff import *
from RecoTracker.TkSeedGenerator.GlobalPixelSeeds_cff import *

# building
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
from RecoTracker.CkfPattern.CkfTrackCandidates_cff import *
newTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone()
newTrajectoryFilter.ComponentName = 'newTrajectoryFilter'
newTrajectoryFilter.filterPset.minimumNumberOfHits = 3
newTrajectoryFilter.filterPset.minPt = 0.3
newTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone()
newTrajectoryBuilder.ComponentName = 'newTrajectoryBuilder'
newTrajectoryBuilder.trajectoryFilterName = 'newTrajectoryFilter'

# fitting
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff import *

### step0
newSeedFromTriplets = RecoTracker.TkSeedGenerator.GlobalSeedsFromTripletsWithVertices_cff.globalSeedsFromTripletsWithVertices.clone()
newSeedFromTriplets.RegionFactoryPSet.RegionPSet.ptMin = 0.5
newTrackCandidateMaker = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
newTrackCandidateMaker.src = cms.InputTag('newSeedFromTriplets')
newTrackCandidateMaker.TrajectoryBuilder = 'newTrajectoryBuilder'
preFilterZeroStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone()
preFilterZeroStepTracks.src = 'newTrackCandidateMaker'
preFilterZeroStepTracks.Fitter = 'KFFittingSmootherWithOutliersRejectionAndRK'
preFilterZeroStepTracks.AlgorithmName = 'ctf'


### step1
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


### quality, merging and final sequence
from RecoTracker.FinalTrackSelectors.TracksWithQuality_cff import *
from RecoTracker.FinalTrackSelectors.MergeTrackCollections_cff import *

firstStep = cms.Sequence(newSeedFromTriplets*newTrackCandidateMaker*preFilterZeroStepTracks*tracksWithQualityZeroStep*
                         zeroStepFilter*newClusters*newPixelRecHits*newStripRecHits*
                         newSeedFromPairs*stepOneTrackCandidateMaker*preFilterStepOneTracks*tracksWithQualityStepOne*
                         firstStepTracksWithQuality)






