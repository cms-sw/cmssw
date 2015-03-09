import FWCore.ParameterSet.Config as cms
# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.DetachedTripletStep_cff

# trajectory seeds
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
detachedTripletStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    simTrackSelection = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.simTrackSelection.clone(
        skipSimTrackIds = [cms.InputTag("initialStepSimTrackIds")],
        pTMin = 0.02,
        maxD0 = 30.0,
        maxZ0 = 50
        ),
    minLayersCrossed = 3,
    originpTMin = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepSeeds.RegionFactoryPSet.RegionPSet.ptMin,
    originHalfLength = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepSeeds.RegionFactoryPSet.RegionPSet.originHalfLength,
    originRadius = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepSeeds.RegionFactoryPSet.RegionPSet.originRadius,
    layerList = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepSeedLayers.layerList.value()
    )

# track candidates
import FastSimulation.Tracking.TrackCandidateProducer_cfi
detachedTripletStepTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    SeedProducer = cms.InputTag("detachedTripletStepSeeds"),
    MinNumberOfCrossedLayers = 3
    )



# tracks 
detachedTripletStepTracks = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepTracks.clone(
    Fitter = 'KFFittingSmootherSecond',
    TTRHBuilder = 'WithoutRefit',
    Propagator = 'PropagatorWithMaterial'

)

# simtrack id producer
detachedTripletStepSimTrackIds = cms.EDProducer("SimTrackIdProducer",
                                  trackCollection = cms.InputTag("detachedTripletStepTracks"),
                                  HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits")
                                  )

#final selection
detachedTripletStepSelector = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepSelector.clone()
detachedTripletStep = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStep.clone() 

# Final sequence 
DetachedTripletStep = cms.Sequence(detachedTripletStepSeeds
                                   +detachedTripletStepTrackCandidates
                                   +detachedTripletStepTracks
                                   +detachedTripletStepSelector
                                   +detachedTripletStep
                                   +detachedTripletStepSimTrackIds
)
