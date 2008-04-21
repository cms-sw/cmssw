import FWCore.ParameterSet.Config as cms

RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    # to save the status of the last event (useful for crashes)
    saveFileName = cms.untracked.string('RandomEngineState.log'),
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(11),
        ecalUnsuppressedDigis = cms.untracked.uint32(1234567),
        muonCSCDigis = cms.untracked.uint32(11223344),
        mix = cms.untracked.uint32(12345),
        siPixelDigis = cms.untracked.uint32(1234567),
        VtxSmeared = cms.untracked.uint32(98765432),
        hcalUnsuppressedDigis = cms.untracked.uint32(11223344),
        muonDTDigis = cms.untracked.uint32(1234567),
        siStripDigis = cms.untracked.uint32(1234567),
        muonRPCDigis = cms.untracked.uint32(1234567)
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


