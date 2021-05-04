import FWCore.ParameterSet.Config as cms
# Low pT Electron value maps
from RecoEgamma.EgammaElectronProducers.lowPtGsfElectronSeedValueMaps_cfi import lowPtGsfElectronSeedValueMaps

# Low pT Electron value maps, rekeyed by reco::GsfElectron
rekeyLowPtGsfElectronSeedValueMaps = lowPtGsfElectronSeedValueMaps.clone(
    rekey=True,
    gsfElectrons="lowPtGsfElectrons",
    floatValueMaps=["lowPtGsfElectronSeedValueMaps:unbiased",
                    "lowPtGsfElectronSeedValueMaps:ptbiased"],
)
