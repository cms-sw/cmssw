import FWCore.ParameterSet.Config as cms

from FastSimulation.Tracking.PixelTripletSeedProducer_cff import *
from FastSimulation.Tracking.PixelTracksProducer_cfi import *
from FastSimulation.Tracking.HLTPixelTracksProducer_cfi import *
hltPixelTracking = cms.Sequence(pixelTripletSeeds+hltPixelTracks)
# Just a copy of the above, for HLT
pixelTracking = cms.Sequence(hltPixelTracking+pixelTracks)
