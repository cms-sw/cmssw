from RecoEgamma.EgammaElectronProducers.ecalDrivenGsfElectronCores_cfi import ecalDrivenGsfElectronCores

ecalDrivenGsfElectronCoresHGC = ecalDrivenGsfElectronCores.clone(
  gsfTracks = 'electronGsfTracks',
  useGsfPfRecTracks = False,
  hgcalOnly = True,
)
# foo bar baz
# uGE6FpU6KpYGU
# UZ3ml48S3Avpt
