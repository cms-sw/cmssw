import FWCore.ParameterSet.Config as cms

from ..modules.hltPhase2PixelFitterByHelixProjections_cfi import *
from ..modules.hltPhase2PixelTrackFilterByKinematics_cfi import *
from ..modules.hltPhase2PixelTracks_cfi import *
from ..modules.hltPhase2PixelTracksHitDoublets_cfi import *
from ..modules.hltPhase2PixelTracksHitSeeds_cfi import *
from ..modules.hltPhase2PixelTracksSeedLayers_cfi import *
from ..modules.hltPhase2PixelTracksAndHighPtStepTrackingRegions_cfi import *

hltPhase2PixelTracksTask = cms.Task(
    hltPhase2PixelFitterByHelixProjections,
    hltPhase2PixelTrackFilterByKinematics,
    hltPhase2PixelTracks,
    hltPhase2PixelTracksHitDoublets,
    hltPhase2PixelTracksHitSeeds,
    hltPhase2PixelTracksSeedLayers,
    hltPhase2PixelTracksAndHighPtStepTrackingRegions
)
