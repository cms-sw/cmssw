import FWCore.ParameterSet.Config as cms

# pixel track producer (with vertex)
from RecoHI.HiTracking.HIPixel3PrimTracks_cfi import *

# pixel seeds
import RecoPixelVertexing.PixelLowPtUtilities.TrackSeeds_cfi
hiPixelTrackSeeds = RecoPixelVertexing.PixelLowPtUtilities.TrackSeeds_cfi.pixelTrackSeeds.clone(
	InputCollection = 'hiPixel3PrimTracks'
  )

hiPrimSeeds = cms.Sequence( PixelLayerTriplets * hiPixel3PrimTracks * hiPixelTrackSeeds )
