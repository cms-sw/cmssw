import FWCore.ParameterSet.Config as cms

from FastSimulation.Tracking.PixelTripletSeedProducer_cff import *
from FastSimulation.Tracking.HLTPixelTracksProducer_cfi import *
hltPixelTracking = cms.Sequence(pixelTripletSeeds+hltPixelTracks)
