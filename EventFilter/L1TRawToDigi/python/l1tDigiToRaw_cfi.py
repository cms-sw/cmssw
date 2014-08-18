import FWCore.ParameterSet.Config as cms

l1tDigiToRaw = cms.EDProducer(
    "l1t::L1TDigiToRaw",
    packers = cms.PSet(
        egamma = cms.PSet(
            type = cms.string("l1t::EGammaPackerFactory"),
            EGammas = cms.InputTag("caloStage1FinalDigis")
            ),
        etsum = cms.PSet(
            type = cms.string("l1t::EtSumPackerFactory"),
            EtSums = cms.InputTag("caloStage1FinalDigis")
            ),
        # calotower = cms.PSet(
        # type = cms.string("l1t::CaloTowerPackerFactory"),
        # CaloTowers = cms.InputTag("Layer2HW")
        # ),
        jet = cms.PSet(
            type = cms.string("l1t::JetPackerFactory"),
            Jets = cms.InputTag("caloStage1FinalDigis")
            ),
        tau = cms.PSet(
            type = cms.string("l1t::TauPackerFactory"),
            Taus = cms.InputTag("caloStage1FinalDigis")
            )
        ),
    InputLabel = cms.InputTag("caloStage1FinalDigis"),
    FedId = cms.int32(100),
    FWId = cms.uint32(1)
)
