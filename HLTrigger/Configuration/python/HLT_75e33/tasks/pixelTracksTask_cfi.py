import FWCore.ParameterSet.Config as cms

from ..modules.hltPhase2PixelFitterByHelixProjections_cfi import *
from ..modules.hltPhase2PixelTrackFilterByKinematics_cfi import *
from ..modules.pixelTracks_cfi import *
from ..modules.hltPhase2PixelTracksHitDoublets_cfi import *
from ..modules.pixelTracksHitSeeds_cfi import *
from ..modules.pixelTracksSeedLayers_cfi import *
from ..modules.hltPhase2PixelTracksAndHighPtStepTrackingRegions_cfi import *

pixelTracksTask = cms.Task(
    hltPhase2PixelFitterByHelixProjections,
    hltPhase2PixelTrackFilterByKinematics,
    pixelTracks,
    hltPhase2PixelTracksHitDoublets,
    pixelTracksHitSeeds,
    pixelTracksSeedLayers,
    hltPhase2PixelTracksAndHighPtStepTrackingRegions
)
