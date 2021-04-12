import FWCore.ParameterSet.Config as cms

from ..modules.pixelFitterByHelixProjections_cfi import *
from ..modules.pixelTrackFilterByKinematics_cfi import *
from ..modules.pixelTracks_cfi import *
from ..modules.pixelTracksHitDoublets_cfi import *
from ..modules.pixelTracksHitSeeds_cfi import *
from ..modules.pixelTracksSeedLayers_cfi import *
from ..modules.pixelTracksTrackingRegions_cfi import *

pixelTracksTask = cms.Task(pixelFitterByHelixProjections, pixelTrackFilterByKinematics, pixelTracks, pixelTracksHitDoublets, pixelTracksHitSeeds, pixelTracksSeedLayers, pixelTracksTrackingRegions)
