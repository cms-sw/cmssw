import FWCore.ParameterSet.Config as cms

import copy
from RecoPixelVertexing.PixelLowPtUtilities.PixelTrackSeeds_cfi import *
#
# Pixel seeds
primSeeds = copy.deepcopy(pixelTrackSeeds)
import copy
from RecoPixelVertexing.PixelLowPtUtilities.PixelTrackSeeds_cfi import *
# Secondary seeds
secoSeeds = copy.deepcopy(pixelTrackSeeds)
primSeeds.tripletList = ['pixel3PrimTracks']
secoSeeds.tripletList = ['pixelSecoTracks']

