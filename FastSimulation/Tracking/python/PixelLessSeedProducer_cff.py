import FWCore.ParameterSet.Config as cms

import FastSimulation.Tracking.TrajectorySeedProducer_cfi
pixelLessSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()

from RecoTracker.TkSeedingLayers.PixelLessLayerPairs_cfi import *

pixelLessSeeds.layerList = PixelLessLayerPairs.layerList
