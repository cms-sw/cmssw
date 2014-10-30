import FWCore.ParameterSet.Config as cms

l1tRawToDigi = cms.EDProducer(
    "l1t::L1TRawToDigi",
    Setup = cms.string("CaloSetup"),
    InputLabel = cms.InputTag("l1tDigiToRaw"),
    FedId = cms.int32(1)
)
