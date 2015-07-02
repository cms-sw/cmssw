import FWCore.ParameterSet.Config as cms

import FastSimulation.Tracking.TrajectorySeedProducer_cfi

globalPixelSeedsForElectrons = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
#globalPixelSeedsForElectrons.firstHitSubDetectorNumber = [2]
#globalPixelSeedsForElectrons.firstHitSubDetectors = [1, 2]
#globalPixelSeedsForElectrons.secondHitSubDetectorNumber = [2]
#globalPixelSeedsForElectrons.secondHitSubDetectors = [1, 2]

#a stripped translation of the old syntax above
globalPixelSeedsForElectrons.layerList = cms.vstring(
    'BPix1+BPix2'
    'BPix1+FPix1_pos',
    'BPix1+FPix1_neg',
    'BPix2+FPix1_pos',
    'BPix2+FPix1_neg',
    
    'FPix1_pos+FPix2_pos',
    'FPix1_neg+FPix2_neg',
)
#globalPixelSeedsForElectrons.simTrackSelection.skipSimTrackIds = [cms.InputTag("globalPixelStepIds")]

#globalPixelSeedsForElectrons.zVertexConstraint = cms.double(0.5)
globalPixelSeedsForElectrons.originRadius = cms.double(0.02)
globalPixelSeedsForElectrons.originpTMin = cms.double(1.5)



globalPixelSeedsForPhotons = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
#globalPixelSeedsForPhotons.firstHitSubDetectorNumber = [2]
#globalPixelSeedsForPhotons.firstHitSubDetectors = [1, 2]
#globalPixelSeedsForPhotons.secondHitSubDetectorNumber = [2]
#globalPixelSeedsForPhotons.secondHitSubDetectors = [1, 2]

#a stripped translation of the old syntax above
globalPixelSeedsForPhotons.layerList = cms.vstring(
    'BPix1+BPix2'
    'BPix1+FPix1_pos',
    'BPix1+FPix1_neg',
    'BPix2+FPix1_pos',
    'BPix2+FPix1_neg',
    
    'FPix1_pos+FPix2_pos',
    'FPix1_neg+FPix2_neg',
)
#globalPixelSeedsForPhotons.simTrackSelection.skipSimTrackIds = [cms.InputTag("globalPixelStepIds")]
globalPixelSeedsForPhotons.originRadius = cms.double(0.02)
globalPixelSeedsForPhotons.originpTMin = cms.double(1.5)

