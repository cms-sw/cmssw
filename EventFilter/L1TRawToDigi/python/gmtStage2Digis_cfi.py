import FWCore.ParameterSet.Config as cms

gmtStage2Digis = cms.EDProducer(
    "L1TRawToDigi",
    Setup = cms.string("stage2::GMTSetup"),
    FedIds = cms.vint32(1402),
)
