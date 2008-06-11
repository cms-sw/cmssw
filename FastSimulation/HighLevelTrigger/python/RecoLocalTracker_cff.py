import FWCore.ParameterSet.Config as cms

from FastSimulation.HighLevelTrigger.DummyModule_cfi import *
from FastSimulation.Tracking.PixelTripletSeedProducer_cff import *
from FastSimulation.Tracking.GlobalPixelSeedProducer_cff import *
pixeltrackerlocalreco = cms.Sequence(pixelTripletSeeds+globalPixelSeeds)
striptrackerlocalreco = cms.Sequence(dummyModule)


