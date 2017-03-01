import FWCore.ParameterSet.Config as cms

# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.PixelPairStep_cff as _standard

# fast tracking mask producer
import FastSimulation.Tracking.FastTrackerRecHitMaskProducer_cfi
pixelPairStepMasks = FastSimulation.Tracking.FastTrackerRecHitMaskProducer_cfi.maskProducerFromClusterRemover(_standard.pixelPairStepClusters)
                               
# tracking regions
pixelPairStepTrackingRegions = _standard.pixelPairStepTrackingRegions.clone(
    RegionPSet=dict(VertexCollection = "firstStepPrimaryVerticesBeforeMixing")
)

# trajectory seeds
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
pixelPairStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    layerList = _standard.pixelPairStepSeedLayers.layerList.value(),
    trackingRegions = "pixelPairStepTrackingRegions",
    hitMasks = cms.InputTag("pixelPairStepMasks"),
)

# track candidate 
import FastSimulation.Tracking.TrackCandidateProducer_cfi
pixelPairStepTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    src = cms.InputTag("pixelPairStepSeeds"),
    MinNumberOfCrossedLayers = 2, # ?
    hitMasks = cms.InputTag("pixelPairStepMasks"),
)

# tracks
pixelPairStepTracks = _standard.pixelPairStepTracks.clone(TTRHBuilder = 'WithoutRefit')

# final Selection
pixelPairStep = _standard.pixelPairStep.clone()
pixelPairStep.vertices = "firstStepPrimaryVerticesBeforeMixing"

# Final sequence 
PixelPairStep = cms.Sequence(pixelPairStepMasks
                             +pixelPairStepTrackingRegions
                             +pixelPairStepSeeds
                             +pixelPairStepTrackCandidates
                             +pixelPairStepTracks
                             +pixelPairStep 
                         )
