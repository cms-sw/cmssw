import FWCore.ParameterSet.Config as cms

l1tRawToDigi = cms.EDProducer(
    "L1TRawToDigi",
    Setup = cms.string("stage2::CaloSetup"),
    InputLabel = cms.InputTag("l1tDigiToRaw"),
    FedId = cms.int32(1),
    FWId = cms.untracked.int32(1)
)
