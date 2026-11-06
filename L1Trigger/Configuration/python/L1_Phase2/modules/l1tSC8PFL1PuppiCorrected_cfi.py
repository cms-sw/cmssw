import FWCore.ParameterSet.Config as cms

l1tSC8PFL1PuppiCorrected = cms.EDProducer("L1SeedConePFJetProducer",
    HW = cms.bool(False),
    L1PFObjects = cms.InputTag("l1tLayer1","Puppi"),
    coneSize = cms.double(0.8),
    correctorDir = cms.string('L1PuppiSC4EmuJets'),
    correctorFile = cms.string('L1Trigger/Phase2L1ParticleFlow/data/jecs/jecs_20220308.root'),
    debug = cms.bool(False),
    doCorrections = cms.bool(True),
    doMass = cms.bool(False),
    mightGet = cms.optional.untracked.vstring,
    nJets = cms.uint32(16),
    sortJets = cms.bool(False),
    wideConeJet = cms.bool(True)
)
