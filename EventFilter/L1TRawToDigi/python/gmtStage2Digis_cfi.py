import FWCore.ParameterSet.Config as cms

gmtStage2Digis = cms.EDProducer(
    "L1TRawToDigi",
    InputLabel = cms.InputTag("rawDataCollector"),
    Setup = cms.string("stage2::GMTSetup"),
    FedIds = cms.vint32(1402),
)
