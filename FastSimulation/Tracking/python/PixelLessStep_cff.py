import FWCore.ParameterSet.Config as cms

# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.PixelLessStep_cff

# simtrack id producer                                                                                                                                                         
import FastSimulation.Tracking.SimTrackIdProducer_cfi
pixelLessStepSimTrackIds=FastSimulation.Tracking.SimTrackIdProducer_cfi.simTrackIdProducer.clone(
    trackCollection = cms.InputTag("mixedTripletStepTracks"),
    TrackQuality = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepClusters.TrackQuality,
    maxChi2 = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepClusters.maxChi2,
    overrideTrkQuals = cms.InputTag('mixedTripletStep')
)

# trajectory seeds 
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
pixelLessStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    simTrackSelection = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.simTrackSelection.clone(
        skipSimTrackIds = [
            cms.InputTag("detachedTripletStepSimTrackIds"),
            cms.InputTag("lowPtTripletStepSimTrackIds"),
            cms.InputTag("pixelPairStepSimTrackIds"),
            cms.InputTag("mixedTripletStepSimTrackIds"),
            cms.InputTag("pixelLessStepSimTrackIds")],
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
PixelLessStep = cms.Sequence(pixelLessStepSimTrackIds
                             +pixelLessStepSeeds
                             +pixelLessStepTrackCandidates
                             +pixelLessStepTracks
                             +pixelLessStepSelector
                             +pixelLessStep                             
                         )

