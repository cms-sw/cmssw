import FWCore.ParameterSet.Config as cms

# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.TobTecStep_cff

# trajectory seeds 
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
tobTecStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    simTrackSelection = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.simTrackSelection.clone(
        skipSimTrackIds = [
            cms.InputTag("initialStepSimTrackIds"),
            cms.InputTag("detachedTripletStepSimTrackIds"),
            cms.InputTag("lowPtTripletStepSimTrackIds"),
            cms.InputTag("pixelPairStepSimTrackIds"),
            cms.InputTag("mixedTripletStepSimTrackIds"),
            cms.InputTag("pixelLessStepSimTrackIds")],
        pTMin = 0.3,
        maxD0 = 99.0,
        maxZ0 = 99
    ),
    minLayersCrossed = 4,
    originpTMin = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepSeedsPair.RegionFactoryPSet.RegionPSet.ptMin,
    originHalfLength = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepSeedsPair.RegionFactoryPSet.RegionPSet.originHalfLength,
    originRadius = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepSeedsPair.RegionFactoryPSet.RegionPSet.originRadius,
    layerList = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepSeedLayersPair.layerList.value()
# only pair seeds
)

# track candidate
import FastSimulation.Tracking.TrackCandidateProducer_cfi
tobTecStepTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    SeedProducer = cms.InputTag("tobTecStepSeeds"),
    MinNumberOfCrossedLayers = 3
)

# tracks 
tobTecStepTracks = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepTracks.clone(
    TTRHBuilder = 'WithoutRefit',
    Fitter = 'KFFittingSmootherFifth',
    Propagator = 'PropagatorWithMaterial'
)

# simtrack id producer
tobTecStepSimTrackIds = cms.EDProducer("SimTrackIdProducer",
                                       trackCollection = cms.InputTag("tobTecStepTracks"),
                                       HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits")
)

# final selection
tobTecStepSelector = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepSelector.clone()
#tobTecStep = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStep.clone()

# Final sequence 
TobTecStep = cms.Sequence(tobTecStepSeeds
                          +tobTecStepTrackCandidates
                          +tobTecStepTracks
                          +tobTecStepSelector                          
                          +tobTecStepSimTrackIds
                      )
