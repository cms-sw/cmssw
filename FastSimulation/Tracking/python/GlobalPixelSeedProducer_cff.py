import FWCore.ParameterSet.Config as cms

import FastSimulation.Tracking.TrajectorySeedProducer_cfi
globalPixelSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
#globalPixelSeeds.firstHitSubDetectorNumber = [2]
#globalPixelSeeds.firstHitSubDetectors = [1, 2]
#globalPixelSeeds.secondHitSubDetectorNumber = [2]
#globalPixelSeeds.secondHitSubDetectors = [1, 2]

#a stripped translation of the old syntax above
globalPixelSeeds.layerList = cms.vstring(
    'BPix1+BPix2'
    'BPix1+FPix1_pos',
    'BPix1+FPix1_neg',
    'BPix2+FPix1_pos',
    'BPix2+FPix1_neg',
    
    'FPix1_pos+FPix2_pos',
    'FPix1_neg+FPix2_neg',
)


