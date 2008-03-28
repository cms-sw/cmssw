import FWCore.ParameterSet.Config as cms

from FastSimulation.Tracking.PixelTripletSeedProducer_cff import *
from FastSimulation.Tracking.PixelTracksProducer_cfi import *
pixelGSTracking = cms.Sequence(pixelTripletGSSeeds+pixelTracks)

