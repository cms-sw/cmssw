import FWCore.ParameterSet.Config as cms

l1tDigiToRaw = cms.EDProducer(
    "l1t::L1TDigiToRaw",
    Packers = cms.vstring([
        # "l1t::CaloTowerPackerFactory",
        "l1t::EGammaPackerFactory",
        "l1t::EtSumPackerFactory",
        "l1t::JetPackerFactory",
        "l1t::TauPackerFactory"
        ]),
    InputLabel = cms.InputTag("caloStage1FinalDigis"),
    FedId = cms.int32(100),
    FWId = cms.uint32(1)
)
