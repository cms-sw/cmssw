import FWCore.ParameterSet.Config as cms

TESTcaloStage1Digis = cms.EDProducer(
    "l1t::L1TRawToDigi",
    Setup = cms.string("stage1::CaloSetup"),
    InputLabel = cms.InputTag("l1tDigiToRaw"),
    #InputLabel = cms.InputTag("rawDataCollector"),
    FedId = cms.int32(1300),
    FWId = cms.untracked.int32(2)
)
