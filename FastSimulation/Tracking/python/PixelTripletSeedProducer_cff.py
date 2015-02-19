import FWCore.ParameterSet.Config as cms

import FastSimulation.Tracking.TrajectorySeedProducer_cfi
pixelTripletSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
pixelTripletSeeds.numberOfHits = 3
pixelTripletSeeds.outputSeedCollectionName = 'PixelTriplet'

from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import PixelLayerTriplets
pixelTripletSeeds.layerList = PixelLayerTriplets.layerList
