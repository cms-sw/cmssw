import FWCore.ParameterSet.Config as cms

l1tSC4PFL1PuppiExtendedCorrectedEmulator = cms.EDProducer("L1SeedConePFJetProducer",
    HW = cms.bool(True),
    L1PFObjects = cms.InputTag("l1tLayer2DeregionizerExtended","Puppi"),
    coneSize = cms.double(0.4),
    correctorDir = cms.string('L1PuppiSC4EmuJets'),
    correctorFile = cms.string('L1Trigger/Phase2L1ParticleFlow/data/jecs/jecs_20220308.root'),
    debug = cms.bool(False),
    doCorrections = cms.bool(True),
    doMass = cms.bool(False),
    mightGet = cms.optional.untracked.vstring,
    nJets = cms.uint32(16),
    sortJets = cms.bool(False),
    wideConeJet = cms.bool(False)
)
