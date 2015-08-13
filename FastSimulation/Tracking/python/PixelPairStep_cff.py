import FWCore.ParameterSet.Config as cms

# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.PixelPairStep_cff

# fast tracking mask producer                                                                                                                                                                                                                                        
from FastSimulation.Tracking.FastTrackingMaskProducer_cfi import fastTrackingMaskProducer as _fastTrackingMaskProducer
pixelPairStepMasks = _fastTrackingMaskProducer.clone(
    trackCollection = cms.InputTag("lowPtTripletStepTracks"),
    TrackQuality = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepClusters.TrackQuality,
    overrideTrkQuals = cms.InputTag('lowPtTripletStep', "QualityMasks"),                        
    oldHitCombinationMasks = cms.InputTag("lowPtTripletStepMasks","hitCombinationMasks"),
    oldHitMasks = cms.InputTag("lowPtTripletStepMasks","hitMasks")
)
  
# trajectory seeds
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
pixelPairStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    simTrackSelection = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.simTrackSelection.clone(
        pTMin = 0,
        maxD0 = -1,
        maxZ0 = -1
        ),
    minLayersCrossed = 2,
layerList = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepSeedLayers.layerList.value(),
    RegionFactoryPSet = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepSeeds.RegionFactoryPSet,
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
'''
    nSigmaZ = 3,
    #hitMasks = cms.InputTag("pixelPairStepMasks","hitMasks"),
    hitCombinationMasks = cms.InputTag("pixelPairStepMasks","hitCombinationMasks"), 
    ptMin = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepSeeds.RegionFactoryPSet.RegionPSet.ptMin,
    originRadius = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepSeeds.RegionFactoryPSet.RegionPSet.originRadius,
    layerList = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepSeedLayers.layerList.value()
'''
)
pixelPairStepSeeds.RegionFactoryPSet.RegionPSet.VertexCollection = cms.InputTag("firstStepPrimaryVerticesBeforeMixing")
# track candidate 
import FastSimulation.Tracking.TrackCandidateProducer_cfi
pixelPairStepTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    src = cms.InputTag("pixelPairStepSeeds"),
    MinNumberOfCrossedLayers = 2 # ?
    #hitMasks = cms.InputTag("pixelPairStepMasks","hitMasks"),
)

# tracks
pixelPairStepTracks = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepTracks.clone(
    Fitter = 'KFFittingSmootherSecond',
    TTRHBuilder = 'WithoutRefit',
    Propagator = 'PropagatorWithMaterial'
)
# final Selection
pixelPairStep = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStep.clone()
pixelPairStep.vertices = "firstStepPrimaryVerticesBeforeMixing"

# Final sequence 
PixelPairStep = cms.Sequence(pixelPairStepMasks
                             +pixelPairStepSeeds
                             +pixelPairStepTrackCandidates
                             +pixelPairStepTracks
                             +pixelPairStep 
                         )
