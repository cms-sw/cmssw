import FWCore.ParameterSet.Config as cms
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
import FastSimulation.Tracking.InitialStep_cff

# pixel triplet seeds
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
hltPixelTripletSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    layerList = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.layerList,
    skipSeedFinderSelector = cms.untracked.bool(True),
    RegionFactoryPSet = FastSimulation.Tracking.InitialStep_cff.initialStepSeeds.RegionFactoryPSet.clone()
    )

# pixel pair seeds
# todo: import layerlist 
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
import RecoTracker.TkSeedingLayers.MixedLayerPairs_cfi
hltPixelPairSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    layerList = RecoTracker.TkSeedingLayers.MixedLayerPairs_cfi.MixedLayerPairs.layerList,
    skipSeedFinderSelector = cms.untracked.bool(True),
    RegionFactoryPSet = FastSimulation.Tracking.InitialStep_cff.initialStepSeeds.RegionFactoryPSet.clone()
    )

# todo: add mixed pair seeds?

hltSeedSequence =cms.Sequence(hltPixelTripletSeeds+hltPixelPairSeeds)
