import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaElectronProducers.defaultLowPtGsfElectronID_cfi import defaultLowPtGsfElectronID
lowPtGsfElectronID = defaultLowPtGsfElectronID.clone()

from Configuration.Eras.Modifier_bParking_cff import bParking
bParking.toModify(
    lowPtGsfElectronID,
    ModelWeights = ["RecoEgamma/ElectronIdentification/data/LowPtElectrons/LowPtElectrons_ID_2021May17.root"],
)
