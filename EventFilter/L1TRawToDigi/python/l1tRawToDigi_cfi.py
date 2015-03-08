import FWCore.ParameterSet.Config as cms

l1tRawToDigi = cms.EDProducer(
    "l1t::L1TRawToDigi",
    Unpackers = cms.vstring([
        "l1t::CaloTowerUnpackerFactory",
        "l1t::EGammaUnpackerFactory",
        "l1t::EtSumUnpackerFactory",
        "l1t::JetUnpackerFactory",
        "l1t::TauUnpackerFactory"
        ]),
    InputLabel = cms.InputTag("l1tDigiToRaw"),
    FedId = cms.int32(100)
)
