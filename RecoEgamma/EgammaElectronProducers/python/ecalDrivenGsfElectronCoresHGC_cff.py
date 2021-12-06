from RecoEgamma.EgammaElectronProducers.ecalDrivenGsfElectronCores_cfi import ecalDrivenGsfElectronCores

ecalDrivenGsfElectronCoresHGC = ecalDrivenGsfElectronCores.clone(
  gsfTracks = 'electronGsfTracks',
  useGsfPfRecTracks = False,
  hgcalOnly = True,
)
