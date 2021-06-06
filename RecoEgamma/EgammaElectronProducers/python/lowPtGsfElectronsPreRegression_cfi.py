from RecoEgamma.EgammaElectronProducers.gsfElectrons_cfi import ecalDrivenGsfElectrons

lowPtGsfElectronsPreRegression = ecalDrivenGsfElectrons.clone(gsfElectronCoresTag = "lowPtGsfElectronCores")

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(lowPtGsfElectronsPreRegression,ctfTracksTag = "generalTracksBeforeMixing")
