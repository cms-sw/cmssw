import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaElectronProducers.lowPtGsfElectronID_cfi import lowPtGsfElectronID

from Configuration.Eras.Modifier_bParking_cff import bParking
bParking.toModify(
    lowPtGsfElectronID,
    ModelWeights = ["RecoEgamma/ElectronIdentification/data/LowPtElectrons/LowPtElectrons_ID_2021May17.root"],
)
