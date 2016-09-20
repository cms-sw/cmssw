import FWCore.ParameterSet.Config as cms

# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.LowPtTripletStep_cff as _standard

# fast tracking mask producer
import FastSimulation.Tracking.FastTrackerRecHitMaskProducer_cfi
lowPtTripletStepMasks = FastSimulation.Tracking.FastTrackerRecHitMaskProducer_cfi.maskProducerFromClusterRemover(_standard.lowPtTripletStepClusters)

# trajectory seeds
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
lowPtTripletStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    layerList = _standard.lowPtTripletStepSeedLayers.layerList.value(),
    RegionFactoryPSet = _standard.lowPtTripletStepSeeds.RegionFactoryPSet,
    hitMasks = cms.InputTag("lowPtTripletStepMasks"),
)
lowPtTripletStepSeeds.seedFinderSelector.pixelTripletGeneratorFactory = _standard.lowPtTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet
#lowPtTripletStepSeeds.pixelTripletGeneratorFactory.SeedComparitorPSet=cms.PSet(  ComponentName = cms.string( "none" ) )

# track candidates
import FastSimulation.Tracking.TrackCandidateProducer_cfi
lowPtTripletStepTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    src = cms.InputTag("lowPtTripletStepSeeds"),
    MinNumberOfCrossedLayers = 3,
    hitMasks = cms.InputTag("lowPtTripletStepMasks"),
)

# tracks
lowPtTripletStepTracks = _standard.lowPtTripletStepTracks.clone(TTRHBuilder = 'WithoutRefit')

# final selection
lowPtTripletStep = _standard.lowPtTripletStep.clone()
lowPtTripletStep.vertices = "firstStepPrimaryVerticesBeforeMixing"

# Final swquence 
LowPtTripletStep = cms.Sequence(lowPtTripletStepMasks
                                +lowPtTripletStepSeeds
                                +lowPtTripletStepTrackCandidates
                                +lowPtTripletStepTracks  
                                +lowPtTripletStep   
                                )
