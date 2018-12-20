import FWCore.ParameterSet.Config as cms

lowPtGsfElectronSeedValueMaps = cms.EDProducer(
    "LowPtGsfElectronSeedValueMapsProducer",
    electrons = cms.InputTag("lowPtGsfElectrons"),
    preIdsValueMap = cms.InputTag("lowPtGsfElectronSeeds"),
    ModelNames = cms.vstring(['unbiased','ptbiased']),
    )
