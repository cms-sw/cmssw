import FWCore.ParameterSet.Config as cms

slimmedPhotons = cms.EDProducer("PATPhotonSlimmer",
    src = cms.InputTag("selectedPatPhotons"),
    linkToPackedPFCandidates = cms.bool(True),
    recoToPFMap = cms.InputTag("particleBasedIsolation","gedPhotons"),
    packedPFCandidates = cms.InputTag("packedPFCandidates"),
)
