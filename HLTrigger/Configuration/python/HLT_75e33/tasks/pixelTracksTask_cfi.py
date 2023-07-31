import FWCore.ParameterSet.Config as cms

from ..modules.hltPhase2PixelFitterByHelixProjections_cfi import *
from ..modules.pixelTrackFilterByKinematics_cfi import *
from ..modules.pixelTracks_cfi import *
from ..modules.pixelTracksHitDoublets_cfi import *
from ..modules.pixelTracksHitSeeds_cfi import *
from ..modules.pixelTracksSeedLayers_cfi import *
from ..modules.pixelTracksTrackingRegions_cfi import *

pixelTracksTask = cms.Task(
    hltPhase2PixelFitterByHelixProjections,
    pixelTrackFilterByKinematics,
    pixelTracks,
    pixelTracksHitDoublets,
    pixelTracksHitSeeds,
    pixelTracksSeedLayers,
    pixelTracksTrackingRegions
)
