import FWCore.ParameterSet.Config as cms

emtfStage2Digis = cms.EDProducer("L1TRawToDigi",
    FWId = cms.uint32(0),
    FedIds = cms.vint32(1384, 1385),
    InputLabel = cms.InputTag("rawDataCollector"),
    MTF7 = cms.untracked.bool(True),
    Setup = cms.string('stage2::EMTFSetup'),
    debug = cms.untracked.bool(False)
)
