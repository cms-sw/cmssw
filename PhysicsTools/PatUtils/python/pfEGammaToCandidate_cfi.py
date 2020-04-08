import FWCore.ParameterSet.Config as cms

pfEGammaToCandidate = cms.EDProducer("PFEGammaToCandidate",
    electrons = cms.InputTag("selectedPatElectrons"),
    photons = cms.InputTag("selectedPatPhotons"),
    electron2pf = cms.InputTag("particleBasedIsolation","gedGsfElectrons"),
    photon2pf = cms.InputTag("particleBasedIsolation","gedPhotons")
)
