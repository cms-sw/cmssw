import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaElectronProducers.defaultLowPtGsfElectronCores_cfi import defaultLowPtGsfElectronCores

lowPtGsfElectronCores = defaultLowPtGsfElectronCores.clone(
    gsfPfRecTracks = cms.InputTag("lowPtGsfElePfGsfTracks"),
    gsfTracks = cms.InputTag("lowPtGsfEleGsfTracks"),
    ctfTracks = cms.InputTag("generalTracks"),
    )

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(lowPtGsfElectronCores,ctfTracks = cms.InputTag("generalTracksBeforeMixing"))
