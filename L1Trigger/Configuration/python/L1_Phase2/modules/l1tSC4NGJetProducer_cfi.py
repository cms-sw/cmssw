import FWCore.ParameterSet.Config as cms

l1tSC4NGJetProducer = cms.EDProducer("L1TSC4NGJetProducer",
    classes = cms.vstring(
        'b',
        'c',
        'uds',
        'g',
        'tau_p',
        'tau_n',
        'mu',
        'e'
    ),
    correctorDir = cms.string(''),
    correctorFile = cms.string(''),
    doJEC = cms.bool(False),
    jets = cms.InputTag("l1tSC4PFL1PuppiExtendedEmulator"),
    l1tSC4NGJetModelPath = cms.string('L1TSC4NGJetModel_v1_0_1'),
    maxEta = cms.double(2.4),
    maxJets = cms.int32(16),
    mightGet = cms.optional.untracked.vstring,
    minPt = cms.double(10),
    nParticles = cms.int32(16),
    returnRawPt = cms.bool(False)
)
