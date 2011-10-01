import FWCore.ParameterSet.Config as cms

whichTracking = 'old' # 'old' is the default for the moment

if(whichTracking=='old'):
    newCombinedSeeds = cms.EDProducer(
        "SeedCombiner",
        seedCollections = cms.VInputTag(
        cms.InputTag('iterativeFirstSeeds','FirstPixelTriplets'),
        cms.InputTag('iterativeFirstSeeds','FirstMixedPairs')
        )
        )
else:
    newCombinedSeeds = cms.EDProducer(
        "SeedCombiner",
        seedCollections = cms.VInputTag(
        cms.InputTag('iterativeInitialSeeds','InitialPixelTriplets'),
        cms.InputTag('iterativeLowPtTripletSeeds','LowPtPixelTriplets'),
        )
        )



