import FWCore.ParameterSet.Config as cms

# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.PixelPairStep_cff

# simtrack id producer                                                                                                                                                         
import FastSimulation.Tracking.SimTrackIdProducer_cfi
pixelPairStepSimTrackIds=FastSimulation.Tracking.SimTrackIdProducer_cfi.simTrackIdProducer.clone(
    trackCollection = cms.InputTag("lowPtTripletStepTracks"),
    TrackQuality = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepClusters.TrackQuality,
    maxChi2 = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepClusters.maxChi2,
    overrideTrkQuals = cms.InputTag('lowPtTripletStepSelector','lowPtTripletStep')
)
# trajectory seeds
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
pixelPairStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    simTrackSelection = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.simTrackSelection.clone(
        skipSimTrackIds = [
            cms.InputTag("detachedTripletStepSimTrackIds"),
            cms.InputTag("lowPtTripletStepSimTrackIds"),
            cms.InputTag("pixelPairStepSimTrackIds")],
#        pTMin = 0.3,
 #       maxD0 = 5.0,
  #      maxZ0 = 50
        pTMin = 0,
        maxD0 = -1,
        maxZ0 = -1,
        ),
    minLayersCrossed = 2,
    layerList = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepSeedLayers.layerList.value(),
    RegionFactoryPSet = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepSeeds.RegionFactoryPSet,
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
)
pixelPairStepSeeds.RegionFactoryPSet.RegionPSet.VertexCollection = cms.InputTag("firstStepPrimaryVerticesBeforeMixing")

# track candidate 
import FastSimulation.Tracking.TrackCandidateProducer_cfi
pixelPairStepTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    SeedProducer = cms.InputTag("pixelPairStepSeeds"),
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
PixelPairStep = cms.Sequence(pixelPairStepSimTrackIds
                             +pixelPairStepSeeds
                             +pixelPairStepTrackCandidates
                             +pixelPairStepTracks
                             +pixelPairStepSelector                                                        
                             )
