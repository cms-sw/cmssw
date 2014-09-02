import FWCore.ParameterSet.Config as cms

import FastSimulation.Tracking.TrajectorySeedProducer_cfi
pixelTripletSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
pixelTripletSeeds.numberOfHits = 3
pixelTripletSeeds.outputSeedCollectionName = 'PixelTriplet'

pixelTripletSeeds.newSyntax = True
#pixelTripletSeeds.layerList = ['BPix1+BPix2+BPix3',
#                               'BPix1+BPix2+FPix1_pos',
#                               'BPix1+BPix2+FPix1_neg',
#                               'BPix1+FPix1_pos+FPix2_pos',
#                               'BPix1+FPix1_neg+FPix2_neg']
from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import PixelLayerTriplets
pixelTripletSeeds.layerList = PixelLayerTriplets.layerList
