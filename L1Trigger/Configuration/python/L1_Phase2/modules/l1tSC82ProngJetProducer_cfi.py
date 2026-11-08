import FWCore.ParameterSet.Config as cms

l1tSC82ProngJetProducer = cms.EDProducer("L1TSC82ProngJetProducer",
    jets = cms.InputTag("l1tSC8PFL1PuppiEmulator"),
    l1tSC82ProngJetModelPath = cms.string('L1TSC82ProngJetModel_v0'),
    maxEta = cms.double(5),
    maxJets = cms.int32(16),
    mightGet = cms.optional.untracked.vstring,
    minPt = cms.double(0),
    nParticles = cms.int32(8)
)
