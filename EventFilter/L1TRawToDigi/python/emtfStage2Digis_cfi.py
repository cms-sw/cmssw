import FWCore.ParameterSet.Config as cms

emtfStage2Digis = cms.EDProducer(
    "L1TRawToDigi",
    Setup = cms.string("stage2::EMTFSetup"),
    InputLabel = cms.InputTag("rawDataCollector"),
    FedIds = cms.vint32(1384, 1385),
    FWId = cms.uint32(0), # Need to implement properly - AWB 23.02.16
    debug = cms.untracked.bool(False),
    MTF7 = cms.untracked.bool(True)
)

