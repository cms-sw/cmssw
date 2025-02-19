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
# Seeding
from RecoTracker.SpecialSeedGenerators.CombinatorialSeedGeneratorForCosmicsRegionalReconstruction_cff import *

# Ckf pattern
import RecoTracker.CkfPattern.CkfTrackCandidatesP5_cff
regionalCosmicCkfTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidatesP5_cff.ckfTrackCandidatesP5.clone(
    src = cms.InputTag( "regionalCosmicTrackerSeeds" ),
)

# Track producer
import RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff
regionalCosmicTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff.ctfWithMaterialTracksCosmics.clone(
    src = cms.InputTag( "regionalCosmicCkfTrackCandidates" ),
)
# Final Sequence
regionalCosmicTracksSeq = cms.Sequence( regionalCosmicTrackerSeeds * regionalCosmicCkfTrackCandidates * regionalCosmicTracks )
