import FWCore.ParameterSet.Config as cms

from FastSimulation.Configuration.simulationSwitch_cff import *
if(simulationType=='phase2'):
    newCombinedSeeds = cms.EDProducer(
        "SeedCombiner",
        seedCollections = cms.VInputTag(
        cms.InputTag('iterativeInitialSeeds','InitialPixelTriplets'),
        )
        )
else:
    newCombinedSeeds = cms.EDProducer(
        "SeedCombiner",
        seedCollections = cms.VInputTag(
        cms.InputTag('iterativeInitialSeeds','InitialPixelTriplets'),
        #    cms.InputTag('iterativeLowPtTripletSeeds','LowPtPixelTriplets'),
        cms.InputTag('iterativePixelPairSeeds','PixelPair'),
        cms.InputTag('iterativeMixedTripletStepSeeds','MixedTriplets'),
        cms.InputTag('iterativePixelLessSeeds','PixelLessPairs'),
        )
        )



