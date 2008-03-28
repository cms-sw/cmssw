import FWCore.ParameterSet.Config as cms

#
# Global tracks
from RecoTracker.CkfPattern.CkfTrackCandidates_cff import *
from RecoPixelVertexing.PixelLowPtUtilities.MinBiasCkfTrajectoryFilterESProducer_cfi import *
from RecoPixelVertexing.PixelLowPtUtilities.TrajectoryCleanerBySharedSeeds_cfi import *
# Final fit
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff import *
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
# Primaries
primTrackCandidates = copy.deepcopy(ckfTrackCandidates)
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
globalPrimTracks = copy.deepcopy(ctfWithMaterialTracks)
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
# Secondaries
secoTrackCandidates = copy.deepcopy(ckfTrackCandidates)
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
globalSecoTracks = copy.deepcopy(ctfWithMaterialTracks)
GroupedCkfTrajectoryBuilder.maxCand = 5
GroupedCkfTrajectoryBuilder.intermediateCleaning = False
GroupedCkfTrajectoryBuilder.alwaysUseInvalidHits = False
GroupedCkfTrajectoryBuilder.trajectoryFilterName = 'MinBiasCkfTrajectoryFilter'
MaterialPropagator.Mass = 0.139
OppositeMaterialPropagator.Mass = 0.139
primTrackCandidates.TrajectoryCleaner = 'TrajectoryCleanerBySharedSeeds'
primTrackCandidates.SeedProducer = 'primSeeds'
primTrackCandidates.RedundantSeedCleaner = 'none'
globalPrimTracks.src = 'primTrackCandidates'
globalPrimTracks.TrajectoryInEvent = True
secoTrackCandidates.TrajectoryCleaner = 'TrajectoryCleanerBySharedSeeds'
secoTrackCandidates.SeedProducer = 'secoSeeds'
secoTrackCandidates.RedundantSeedCleaner = 'none'
globalSecoTracks.src = 'secoTrackCandidates'
globalSecoTracks.TrajectoryInEvent = True

