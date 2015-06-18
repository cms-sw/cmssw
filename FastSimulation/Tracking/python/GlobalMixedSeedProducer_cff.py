import FWCore.ParameterSet.Config as cms

import FastSimulation.Tracking.TrajectorySeedProducer_cfi
globalMixedSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
#globalMixedSeeds.firstHitSubDetectorNumber = [3]
#globalMixedSeeds.firstHitSubDetectors = [1, 2, 6]
#globalMixedSeeds.secondHitSubDetectorNumber = [3]
#globalMixedSeeds.secondHitSubDetectors = [1, 2, 6]

#a stripped translation of the old syntax above
globalMixedSeeds.layerList = cms.vstring(
    'BPix1+BPix2'
    'BPix1+FPix1_pos',
    'BPix1+FPix1_neg',
    'BPix2+FPix1_pos',
    'BPix2+FPix1_neg',
    
    'FPix1_pos+FPix2_pos',
    'FPix1_neg+FPix2_neg',
    
    'FPix1_pos+TEC1_pos',
    'FPix1_neg+TEC2_neg',
    
    'FPix2_pos+TEC1_pos',
    'FPix2_neg+TEC2_neg',
)


