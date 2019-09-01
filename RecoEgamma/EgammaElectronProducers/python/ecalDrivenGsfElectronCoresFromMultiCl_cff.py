from RecoEgamma.EgammaElectronProducers.ecalDrivenGsfElectronCores_cfi import ecalDrivenGsfElectronCores

ecalDrivenGsfElectronCoresFromMultiCl = ecalDrivenGsfElectronCores.clone(
  gsfTracks = 'electronGsfTracksFromMultiCl',
  useGsfPfRecTracks = False
)
