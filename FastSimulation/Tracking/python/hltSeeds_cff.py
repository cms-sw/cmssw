import FWCore.ParameterSet.Config as cms
import FastSimulation.Tracking.TrajectorySeedProducer_cfi

# pixel triplet seeds
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
hltPixelTripletSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    layerList = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.layerList
    )

# pixel pair seeds
# todo: import layerlist 
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
hltPixelPairSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    layerList = cms.vstring(
        'BPix1+BPix2'
        'BPix1+FPix1_pos',
        'BPix1+FPix1_neg',
        'BPix2+FPix1_pos',
        'BPix2+FPix1_neg',
        
        'FPix1_pos+FPix2_pos',
        'FPix1_neg+FPix2_neg',
        )
    )

# todo: add mixed pair seeds?

hltSeedSequence =cms.Sequence(hltPixelTripletSeeds+hltPixelPairSeeds)
