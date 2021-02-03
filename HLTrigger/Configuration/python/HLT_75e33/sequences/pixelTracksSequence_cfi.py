import FWCore.ParameterSet.Config as cms

from ..modules.pixelFitterByHelixProjections_cfi import *
from ..modules.pixelTrackFilterByKinematics_cfi import *
from ..modules.pixelTracks_cfi import *
from ..modules.pixelTracksHitDoublets_cfi import *
from ..modules.pixelTracksHitSeeds_cfi import *
from ..modules.pixelTracksSeedLayers_cfi import *
from ..modules.pixelTracksTrackingRegions_cfi import *

pixelTracksSequence = cms.Sequence(pixelTrackFilterByKinematics+pixelFitterByHelixProjections+pixelTracksTrackingRegions+pixelTracksSeedLayers+pixelTracksHitDoublets+pixelTracksHitSeeds+pixelTracks)
