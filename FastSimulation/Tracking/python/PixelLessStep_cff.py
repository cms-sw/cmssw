import FWCore.ParameterSet.Config as cms

# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.PixelLessStep_cff

# fast tracking mask producer                                                                                                                                                         
import FastSimulation.Tracking.FastTrackingMaskProducer_cfi
pixelLessStepFastTrackingMasks=FastSimulation.Tracking.FastTrackingMaskProducer_cfi.fastTrackingMaskProducer.clone(
    trackCollection = cms.InputTag("mixedTripletStepTracks"),
    TrackQuality = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepClusters.TrackQuality,
    overrideTrkQuals = cms.InputTag('mixedTripletStep')
)

# trajectory seeds 
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
pixelLessStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    simTrackSelection = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.simTrackSelection.clone(
       pTMin = 0.3,
        maxD0 = -1,
        maxZ0 = -1
        ),
    minLayersCrossed = 3,
    ptMin = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepSeeds.RegionFactoryPSet.RegionPSet.ptMin,
    originHalfLength = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepSeeds.RegionFactoryPSet.RegionPSet.originHalfLength,
    originRadius = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepSeeds.RegionFactoryPSet.RegionPSet.originRadius,
    layerList = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepSeedLayers.layerList.value()
)

# track candidates
import FastSimulation.Tracking.TrackCandidateProducer_cfi
pixelLessStepTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    src = cms.InputTag("pixelLessStepSeeds"),
    MinNumberOfCrossedLayers = 6 # ?
)

# tracks
pixelLessStepTracks = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepTracks.clone(
    TTRHBuilder = 'WithoutRefit',
    Fitter = 'KFFittingSmootherFourth',
    Propagator = 'PropagatorWithMaterial'
)
# final selection
pixelLessStepSelector = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepSelector.clone()
pixelLessStepSelector.vertices = "firstStepPrimaryVerticesBeforeMixing"
pixelLessStep = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStep.clone()

# Final sequence 
PixelLessStep = cms.Sequence(pixelLessStepFastTrackingMasks
                             +pixelLessStepSeeds
                             +pixelLessStepTrackCandidates
                             +pixelLessStepTracks
                             +pixelLessStepSelector
                             +pixelLessStep                             
                         )

