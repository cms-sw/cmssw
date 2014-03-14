import FWCore.ParameterSet.Config as cms

import FastSimulation.Tracking.TrajectorySeedProducer_cfi
pixelTripletSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
pixelTripletSeeds.numberOfHits = [3]
pixelTripletSeeds.firstHitSubDetectorNumber = [2]
pixelTripletSeeds.firstHitSubDetectors = [1, 2]
pixelTripletSeeds.secondHitSubDetectorNumber = [2]
pixelTripletSeeds.secondHitSubDetectors = [1, 2]
pixelTripletSeeds.thirdHitSubDetectorNumber = [2]
pixelTripletSeeds.thirdHitSubDetectors = [1, 2]
pixelTripletSeeds.seedingAlgo = ['PixelTriplet']

pixelTripletSeeds.newSyntax = True
#pixelTripletSeeds.layerList = ['BPix1+BPix2+BPix3',
#                               'BPix1+BPix2+FPix1_pos',
#                               'BPix1+BPix2+FPix1_neg',
#                               'BPix1+FPix1_pos+FPix2_pos',
#                               'BPix1+FPix1_neg+FPix2_neg']
from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import PixelLayerTriplets
pixelTripletSeeds.layerList = PixelLayerTriplets.layerList
