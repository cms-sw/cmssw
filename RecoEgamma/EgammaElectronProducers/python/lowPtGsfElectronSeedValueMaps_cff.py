import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaElectronProducers.defaultLowPtGsfElectronSeedValueMaps_cfi import defaultLowPtGsfElectronSeedValueMaps

lowPtGsfElectronSeedValueMaps = defaultLowPtGsfElectronSeedValueMaps.clone(
    ModelNames = cms.vstring(['unbiased','ptbiased'])
    )
