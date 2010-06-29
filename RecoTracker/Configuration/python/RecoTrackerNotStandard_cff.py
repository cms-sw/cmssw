import FWCore.ParameterSet.Config as cms

#
# Tracker Tracking etc
#
# Seeds 
from RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cff import *
from RecoTracker.TkSeedGenerator.GlobalSeedsFromPairsWithVertices_cff import *
from RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff import *
from RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff import *
from RecoTracker.TkSeedGenerator.GlobalPixelSeeds_cff import *
from RecoTracker.TkSeedGenerator.GlobalPixelLessSeeds_cff import *
# Ckf
from RecoTracker.CkfPattern.CkfTrackCandidatesNoOverlaps_cff import *
from RecoTracker.CkfPattern.CkfTrackCandidatesPixelLess_cff import *
from RecoTracker.CkfPattern.CkfTrackCandidatesCombinedSeeds_cff import *
# Final Fit
from RecoTracker.TrackProducer.CTFNoOverlaps_cff import *
from RecoTracker.TrackProducer.CTFPixelLess_cff import *
from RecoTracker.TrackProducer.CTFCombinedSeeds_cff import *
ctfTracksNoOverlaps = cms.Sequence(ckfTrackCandidatesNoOverlaps*ctfNoOverlaps)
ctfTracksPixelLess = cms.Sequence(globalPixelLessSeeds*ckfTrackCandidatesPixelLess*ctfPixelLess)
ctfTracksCombinedSeeds = cms.Sequence(globalSeedsFromPairsWithVertices*globalSeedsFromTriplets*globalCombinedSeeds*ckfTrackCandidatesCombinedSeeds*ctfCombinedSeeds)

#
# Regional reconstruction for cosmics
#
# Seeds
from RecoTracker.SpecialSeedGenerators.CombinatorialSeedGeneratorForCosmicsRegionalReconstruction_cff import *
# Ckf
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
regionalCosmicCkfTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag( "regionalCosmicTrackerSeeds" ),
    NavigationSchool = cms.string('CosmicNavigationSchool'),
    #TrajectoryBuilder = cms.string( "CkfTrajectoryBuilder" ),
    TrajectoryBuilder = cms.string( "GroupedCkfTrajectoryBuilder" ),
)

# Track producer
import RecoTracker.TrackProducer.TrackProducer_cfi
regionalCosmicTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = cms.InputTag( "regionalCosmicCkfTrackCandidates" ),
    NavigationSchool = 'CosmicNavigationSchool',
    AlgorithmName = 'cosmics',
    alias = 'regionalCosmicTracks'
)
# Final Sequence
regionalCosmicTracksSeq = cms.Sequence( regionalCosmicTrackerSeeds * regionalCosmicCkfTrackCandidates * regionalCosmicTracks )
