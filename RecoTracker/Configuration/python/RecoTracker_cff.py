import FWCore.ParameterSet.Config as cms

#
# Tracker Tracking
#
# general CPEs 
from RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff import *
# TTRHBuilders
from RecoTracker.TransientTrackingRecHit.TTRHBuilders_cff import *
# Seeds 
from RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff import *
from RecoTracker.TkSeedGenerator.GlobalPixelSeeds_cff import *
# Ckf
from RecoTracker.CkfPattern.CkfTrackCandidates_cff import *
# Final Fit
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff import *
# RoadSearchSeedFinder
from RecoTracker.RoadSearchSeedFinder.RoadSearchSeeds_cff import *
# RoadSearchCloudMaker
from RecoTracker.RoadSearchCloudMaker.RoadSearchClouds_cff import *
# RoadSearchTrackCandidateMaker
from RecoTracker.RoadSearchTrackCandidateMaker.RoadSearchTrackCandidates_cff import *
# RS track fit with material 
from RecoTracker.TrackProducer.RSFinalFitWithMaterial_cff import *
# Iterative Tracking
from RecoTracker.IterativeTracking.iterativeTk_cff import *
# new tracking configuration ################
# it is placed here as a temporary solution
# Seeding 
from RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cff import *
from RecoTracker.TkSeedGenerator.GlobalSeedsFromPairsWithVertices_cff import *
from RecoTracker.TkSeedGenerator.GlobalSeedsFromTripletsWithVertices_cff import *
import RecoTracker.TkSeedGenerator.GlobalSeedsFromPairsWithVertices_cfi
newSeedFromPairs = RecoTracker.TkSeedGenerator.GlobalSeedsFromPairsWithVertices_cfi.globalSeedsFromPairsWithVertices.clone()
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTripletsWithVertices_cfi
newSeedFromTriplets = RecoTracker.TkSeedGenerator.GlobalSeedsFromTripletsWithVertices_cfi.globalSeedsFromTripletsWithVertices.clone()
import RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cfi
newCombinedSeeds = RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cfi.globalCombinedSeeds.clone()
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
# Building
newTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone()
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
newTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone()
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
newTrackCandidateMaker = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
import TrackingTools.TrackFitters.RungeKuttaKFFittingSmootherESProducer_cfi
# Final Fitting
FittingSmootherWithOutlierRejection = TrackingTools.TrackFitters.RungeKuttaKFFittingSmootherESProducer_cfi.RKFittingSmoother.clone()
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
preFilterFirstStepTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
# Collection cleaning and quality
from RecoTracker.FinalTrackSelectors.TracksWithQuality_cff import *
# defines sequence tracksWithQuality, input is preFilterFirstStepTracks, output is firstStepTracksWithQuality
#
# include track colleciton merging sequence
from RecoTracker.FinalTrackSelectors.MergeTrackCollections_cff import *
#
#sequence ckftracks = {globalMixedSeeds,globalPixelSeeds, ckfTrackCandidates,ctfWithMaterialTracks} #only old ctf sequence
newTracking = cms.Sequence(newSeedFromPairs*newSeedFromTriplets*newCombinedSeeds*newTrackCandidateMaker*preFilterFirstStepTracks*tracksWithQuality)
ckftracks = cms.Sequence(newTracking*iterTracking*trackCollectionMerging)
rstracks = cms.Sequence(roadSearchSeeds*roadSearchClouds*rsTrackCandidates*rsWithMaterialTracks)
newSeedFromPairs.RegionFactoryPSet.RegionPSet.ptMin = 0.9
newSeedFromTriplets.RegionFactoryPSet.RegionPSet.ptMin = 0.5
newCombinedSeeds.PairCollection = 'newSeedFromPairs'
newCombinedSeeds.TripletCollection = 'newSeedFromTriplets'
newTrajectoryFilter.ComponentName = 'newTrajectoryFilter'
newTrajectoryFilter.filterPset.minimumNumberOfHits = 3
newTrajectoryFilter.filterPset.minPt = 0.3
newTrajectoryBuilder.ComponentName = 'newTrajectoryBuilder'
newTrajectoryBuilder.trajectoryFilterName = 'newTrajectoryFilter'
newTrackCandidateMaker.SeedProducer = 'newCombinedSeeds'
newTrackCandidateMaker.TrajectoryBuilder = 'newTrajectoryBuilder'
newTrackCandidateMaker.useHitsSplitting = True
newTrackCandidateMaker.doSeedingRegionRebuilding = True
FittingSmootherWithOutlierRejection.ComponentName = 'FittingSmootherWithOutlierRejection'
FittingSmootherWithOutlierRejection.EstimateCut = 20
FittingSmootherWithOutlierRejection.MinNumberOfHits = 3
preFilterFirstStepTracks.src = 'newTrackCandidateMaker'
preFilterFirstStepTracks.TTRHBuilder = 'WithAngleAndTemplate'
preFilterFirstStepTracks.Fitter = 'FittingSmootherWithOutlierRejection'
preFilterFirstStepTracks.AlgorithmName = 'ctf'

