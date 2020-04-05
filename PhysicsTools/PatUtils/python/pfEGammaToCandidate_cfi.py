import FWCore.ParameterSet.Config as cms

pfEGammaToCandidate = cms.EDProducer("PFEGammaToCandidate",
    electrons = cms.InputTag("selectedPatElectrons"),
    photons = cms.InputTag("selectedPatPhotons"),
    electron2pf = cms.InputTag("reducedEgamma","reducedGsfElectronPfCandMap"),
    photon2pf = cms.InputTag("reducedEgamma","reducedPhotonPfCandMap")
)
