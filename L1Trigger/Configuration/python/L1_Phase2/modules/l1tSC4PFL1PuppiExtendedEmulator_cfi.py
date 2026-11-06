import FWCore.ParameterSet.Config as cms

l1tSC4PFL1PuppiExtendedEmulator = cms.EDProducer("L1SeedConePFJetProducer",
    HW = cms.bool(True),
    L1PFObjects = cms.InputTag("l1tLayer2DeregionizerExtended","Puppi"),
    coneSize = cms.double(0.4),
    correctorDir = cms.string(''),
    correctorFile = cms.string(''),
    debug = cms.bool(False),
    doCorrections = cms.bool(False),
    doMass = cms.bool(False),
    mightGet = cms.optional.untracked.vstring,
    nJets = cms.uint32(16),
    sortJets = cms.bool(False),
    wideConeJet = cms.bool(False)
)
