import FWCore.ParameterSet.Config as cms

RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    # to save the status of the last event (useful for crashes)
    saveFileName = cms.untracked.string(''),
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(11),
        simEcalUnsuppressedDigis = cms.untracked.uint32(1234567),
        simMuonCSCDigis = cms.untracked.uint32(11223344),
        mix = cms.untracked.uint32(12345),
        simSiPixelDigis = cms.untracked.uint32(1234567),
        VtxSmeared = cms.untracked.uint32(98765432),
        simHcalUnsuppressedDigis = cms.untracked.uint32(11223344),
        simMuonDTDigis = cms.untracked.uint32(1234567),
        simSiStripDigis = cms.untracked.uint32(1234567),
        simMuonRPCDigis = cms.untracked.uint32(1234567)
    ),
    # to restore the status of the last event, 
    # comment the line above and decomment the following one
    #   untracked string restoreFileName = "RandomEngineState.log"  
    # to reproduce events using the RandomEngineStateProducer (source excluded),
    # comment the sourceSeed definition, decomment the following one
    #   untracked string restoreStateLabel = "randomEngineStateProducer"
    sourceSeed = cms.untracked.uint32(123456789)
)

randomEngineStateProducer = cms.EDProducer("RandomEngineStateProducer")


