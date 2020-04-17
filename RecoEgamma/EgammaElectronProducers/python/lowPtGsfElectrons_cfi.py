from RecoEgamma.EgammaElectronProducers.gsfElectrons_cfi import ecalDrivenGsfElectrons

lowPtGsfElectrons = ecalDrivenGsfElectrons.clone(gsfElectronCoresTag = "lowPtGsfElectronCores")

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(lowPtGsfElectrons,ctfTracksTag = "generalTracksBeforeMixing")

from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
pp_on_AA_2018.toModify(lowPtGsfElectrons.preselection, minSCEtBarrel = 15.0)
pp_on_AA_2018.toModify(lowPtGsfElectrons.preselection, minSCEtEndcaps = 15.0)
