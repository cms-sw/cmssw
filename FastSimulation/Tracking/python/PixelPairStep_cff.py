import FWCore.ParameterSet.Config as cms

# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.PixelPairStep_cff

# fast tracking mask producer                                                                                                                                                                                                                                        
from FastSimulation.Tracking.FastTrackingMaskProducer_cfi import fastTrackingMaskProducer as _fastTrackingMaskProducer
pixelPairStepFastTrackingMasks = _fastTrackingMaskProducer.clone(
    trackCollection = cms.InputTag("lowPtTripletStepTracks"),
    TrackQuality = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepClusters.TrackQuality,
    overrideTrkQuals = cms.InputTag('lowPtTripletStepSelector','lowPtTripletStep')
)
  
# trajectory seeds
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
pixelPairStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    simTrackSelection = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.simTrackSelection.clone(
        pTMin = 0.3,
        maxD0 = 5.0,
        maxZ0 = 50
        ),
    minLayersCrossed = 2,
    nSigmaZ = 3,
    hitMasks = cms.InputTag("pixelPairStepFastTrackingMasks","hitMasks"),
    hitCombinationMasks = cms.InputTag("pixelPairStepFastTrackingMasks","hitCombinationMasks"), 
    ptMin = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepSeeds.RegionFactoryPSet.RegionPSet.ptMin,
    originRadius = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepSeeds.RegionFactoryPSet.RegionPSet.originRadius,
    layerList = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepSeedLayers.layerList.value()
)

# track candidate 
import FastSimulation.Tracking.TrackCandidateProducer_cfi
pixelPairStepTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    src = cms.InputTag("pixelPairStepSeeds"),
    MinNumberOfCrossedLayers = 2 # ?
)

# tracks
pixelPairStepTracks = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepTracks.clone(
    Fitter = 'KFFittingSmootherSecond',
    TTRHBuilder = 'WithoutRefit',
    Propagator = 'PropagatorWithMaterial'
)
# final Selection
pixelPairStepSelector = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepSelector.clone()
pixelPairStepSelector.vertices = "firstStepPrimaryVerticesBeforeMixing"

# Final sequence 
PixelPairStep = cms.Sequence(pixelPairStepFastTrackingMasks
                             +pixelPairStepSeeds
                             +pixelPairStepTrackCandidates
                             +pixelPairStepTracks
                             +pixelPairStepSelector                                                        
                         )
