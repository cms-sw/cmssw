import FWCore.ParameterSet.Config as cms
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
import FastSimulation.Tracking.InitialStep_cff

# tracking regions
hltPixelTripletTrackingRegions = FastSimulation.Tracking.InitialStep_cff.initialStepTrackingRegions.clone()

# pixel triplet seeds
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
hltPixelTripletSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    skipSeedFinderSelector = cms.untracked.bool(True),
    trackingRegions = "hltPixelTripletTrackingRegions"
    )

# pixel pair seeds
# todo: import layerlist 
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
import RecoTracker.TkSeedingLayers.MixedLayerPairs_cfi
hltPixelPairSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    skipSeedFinderSelector = cms.untracked.bool(True),
    trackingRegions = "hltPixelTripletTrackingRegions"
    )

# todo: add mixed pair seeds?

hltSeedSequence =cms.Sequence(hltPixelTripletTrackingRegions+hltPixelTripletSeeds+hltPixelPairSeeds)
