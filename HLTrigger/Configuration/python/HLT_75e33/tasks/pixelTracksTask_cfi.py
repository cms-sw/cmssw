import FWCore.ParameterSet.Config as cms

from ..modules.hltPhase2PixelFitterByHelixProjections_cfi import *
from ..modules.hltPhase2PixelTrackFilterByKinematics_cfi import *
from ..modules.pixelTracks_cfi import *
from ..modules.pixelTracksHitDoublets_cfi import *
from ..modules.pixelTracksHitSeeds_cfi import *
from ..modules.pixelTracksSeedLayers_cfi import *
from ..modules.pixelTracksTrackingRegions_cfi import *

pixelTracksTask = cms.Task(
    hltPhase2PixelFitterByHelixProjections,
    hltPhase2PixelTrackFilterByKinematics,
    pixelTracks,
    pixelTracksHitDoublets,
    pixelTracksHitSeeds,
    pixelTracksSeedLayers,
    pixelTracksTrackingRegions
)
