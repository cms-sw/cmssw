import FWCore.ParameterSet.Config as cms

l1tSC8PFL1PuppiEmulator = cms.EDProducer("L1SeedConePFJetProducer",
    HW = cms.bool(True),
    L1PFObjects = cms.InputTag("l1tLayer2Deregionizer","Puppi"),
    coneSize = cms.double(0.8),
    correctorDir = cms.string(''),
    correctorFile = cms.string(''),
    debug = cms.bool(False),
    doCorrections = cms.bool(False),
    doMass = cms.bool(False),
    mightGet = cms.optional.untracked.vstring,
    nJets = cms.uint32(16),
    sortJets = cms.bool(False),
    wideConeJet = cms.bool(True)
)
