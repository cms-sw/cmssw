import FWCore.ParameterSet.Config as cms

RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    theSource = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    generator = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    evtgenproducer = cms.PSet( 
        initialSeed = cms.untracked.uint32(93278151),
        engineName = cms.untracked.string('HepJamesRandom') 
    ),
    VtxSmeared = cms.PSet(
        initialSeed = cms.untracked.uint32(98765432),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    LHCTransport = cms.PSet(
        initialSeed = cms.untracked.uint32(87654321),
        engineName = cms.untracked.string('TRandom3')
    ),
    g4SimHits = cms.PSet(
        initialSeed = cms.untracked.uint32(11),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    mix = cms.PSet(
        initialSeed = cms.untracked.uint32(12345),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    simSiStripDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(1234567),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    simSiPixelDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(1234567),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    simEcalUnsuppressedDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(1234567),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    simHcalUnsuppressedDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(11223344),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    simMuonDTDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(1234567),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    simMuonCSCDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(11223344),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    simMuonRPCDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(1234567),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    simCastorDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(12345678),
        engineName = cms.untracked.string('HepJamesRandom')
    )
    # to save the status of the last event (useful for crashes)
    ,saveFileName = cms.untracked.string('')
    # to restore the status of the last event, 
    # comment the line above and decomment the following one
    #   ,restoreFileName = cms.untracked.string('RandomEngineState.log')  
    # to reproduce events using the RandomEngineStateProducer (source excluded),
    # comment the sourceSeed definition, decomment the following one
    #   ,restoreStateLabel = cms.untracked.string('randomEngineStateProducer')
)

randomEngineStateProducer = cms.EDProducer("RandomEngineStateProducer")


