import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaElectronProducers.defaultLowPtGsfElectronID_cfi import defaultLowPtGsfElectronID
lowPtGsfElectronID = defaultLowPtGsfElectronID.clone()

from Configuration.Eras.Modifier_bParking_cff import bParking
bParking.toModify(
    lowPtGsfElectronID,
    ModelWeights = ["RecoEgamma/ElectronIdentification/data/LowPtElectrons/LowPtElectrons_ID_2021May17.root"],
)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
from Configuration.ProcessModifiers.run2_miniAOD_UL_cff import run2_miniAOD_UL
from PhysicsTools.NanoAOD.nano_eras_cff import *
(fastSim & (run2_miniAOD_UL | run2_nanoAOD_106Xv2)).toModify(
    lowPtGsfElectronID,
    useGsfToTrack = True,
)
