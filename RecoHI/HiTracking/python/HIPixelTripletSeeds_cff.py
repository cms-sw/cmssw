import FWCore.ParameterSet.Config as cms

# pixel track producer (with vertex)
from RecoHI.HiTracking.HIPixel3PrimTracks_cfi import *
from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import *

# pixel seeds
import RecoTracker.PixelLowPtUtilities.TrackSeeds_cfi
hiPixelTrackSeeds = RecoTracker.PixelLowPtUtilities.TrackSeeds_cfi.pixelTrackSeeds.clone(
	InputCollection = 'hiPixel3PrimTracks'
  )

hiPrimSeedsTask = cms.Task( PixelLayerTriplets , hiPixel3PrimTracksTask , hiPixelTrackSeeds )
hiPrimSeeds = cms.Sequence(hiPrimSeedsTask)
