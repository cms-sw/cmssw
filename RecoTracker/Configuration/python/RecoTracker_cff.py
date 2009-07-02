import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff import *
from RecoTracker.TransientTrackingRecHit.TTRHBuilders_cff import *


# seeding
from RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cff import *
from RecoTracker.TkSeedGenerator.GlobalSeedsFromPairsWithVertices_cff import *
from RecoTracker.TkSeedGenerator.GlobalSeedsFromTripletsWithVertices_cff import *
from RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff import *
from RecoTracker.TkSeedGenerator.GlobalPixelSeeds_cff import *

import RecoTracker.TkSeedGenerator.GlobalSeedsFromPairsWithVertices_cfi
newSeedFromPairs = RecoTracker.TkSeedGenerator.GlobalSeedsFromPairsWithVertices_cfi.globalSeedsFromPairsWithVertices.clone()
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTripletsWithVertices_cfi
newSeedFromTriplets = RecoTracker.TkSeedGenerator.GlobalSeedsFromTripletsWithVertices_cfi.globalSeedsFromTripletsWithVertices.clone()
import RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cfi
newCombinedSeeds = RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cfi.globalCombinedSeeds.clone()

newSeedFromPairs.RegionFactoryPSet.RegionPSet.ptMin = 0.9
newSeedFromTriplets.RegionFactoryPSet.RegionPSet.ptMin = 0.5
newCombinedSeeds.PairCollection = 'newSeedFromPairs'
newCombinedSeeds.TripletCollection = 'newSeedFromTriplets'

# building
from RecoTracker.RoadSearchTrackCandidateMaker.RoadSearchTrackCandidates_cff import *
from RecoTracker.CkfPattern.CkfTrackCandidates_cff import *

import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
newTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone()

import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
newTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone()
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
newTrackCandidateMaker = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()

newTrajectoryFilter.ComponentName = 'newTrajectoryFilter'
newTrajectoryFilter.filterPset.minimumNumberOfHits = 3
newTrajectoryFilter.filterPset.minPt = 0.3

newTrajectoryBuilder.ComponentName = 'newTrajectoryBuilder'
newTrajectoryBuilder.trajectoryFilterName = 'newTrajectoryFilter'

newTrackCandidateMaker.SeedProducer = 'newCombinedSeeds'
newTrackCandidateMaker.TrajectoryBuilder = 'newTrajectoryBuilder'

# fitting
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff import *

import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
preFilterFirstStepTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
preFilterFirstStepTracks.src = 'newTrackCandidateMaker'
preFilterFirstStepTracks.Fitter = 'KFFittingSmootherWithOutliersRejectionAndRK'
preFilterFirstStepTracks.AlgorithmName = 'ctf'

# Iterative steps
from RecoTracker.IterativeTracking.iterativeTk_cff import *


# RS
from RecoTracker.RoadSearchSeedFinder.RoadSearchSeeds_cff import *
from RecoTracker.RoadSearchCloudMaker.RoadSearchClouds_cff import *
from RecoTracker.TrackProducer.RSFinalFitWithMaterial_cff import *


# track collection filtering
from RecoTracker.FinalTrackSelectors.TracksWithQuality_cff import *
from RecoTracker.FinalTrackSelectors.MergeTrackCollections_cff import *


newTracking = cms.Sequence(newSeedFromPairs*newSeedFromTriplets*newCombinedSeeds*
                           newTrackCandidateMaker*
                           preFilterFirstStepTracks*
                           tracksWithQuality)

ckftracks = cms.Sequence(newTracking*
                         iterTracking*
                         trackCollectionMerging)

rstracks = cms.Sequence(roadSearchSeeds*
                        roadSearchClouds*rsTrackCandidates*
                        rsWithMaterialTracks)



