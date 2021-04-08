import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaElectronProducers.lowPtGsfElectronCores_cfi import lowPtGsfElectronCores

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(lowPtGsfElectronCores,ctfTracks = "generalTracksBeforeMixing")
