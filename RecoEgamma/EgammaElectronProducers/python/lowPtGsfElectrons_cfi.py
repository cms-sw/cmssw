from RecoEgamma.EgammaElectronProducers.gsfElectrons_cfi import ecalDrivenGsfElectrons

lowPtGsfElectronsPreRegression = ecalDrivenGsfElectrons.clone(gsfElectronCoresTag = "lowPtGsfElectronCores")

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(lowPtGsfElectronsPreRegression,ctfTracksTag = "generalTracksBeforeMixing")

from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toModify(lowPtGsfElectronsPreRegression.preselection, minSCEtBarrel = 15.0)
pp_on_AA.toModify(lowPtGsfElectronsPreRegression.preselection, minSCEtEndcaps = 15.0)
