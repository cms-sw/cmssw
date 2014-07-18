import FWCore.ParameterSet.Config as cms

l1tDigiToRaw = cms.EDProducer(
    "l1t::L1TDigiToRaw",
    packers = cms.PSet(
        egamma = cms.PSet(
            type = cms.string("l1t::EGammaPackerFactory"),
            EGammas = cms.InputTag("Layer2HW")
            ),
        etsum = cms.PSet(
            type = cms.string("l1t::EtSumPackerFactory"),
            EtSums = cms.InputTag("Layer2HW")
            ),
        # calotower = cms.PSet(
        # type = cms.string("l1t::CaloTowerPackerFactory"),
        # CaloTowers = cms.InputTag("Layer2HW")
        # ),
        jet = cms.PSet(
            type = cms.string("l1t::JetPackerFactory"),
            Jets = cms.InputTag("Layer2HW")
            ),
        tau = cms.PSet(
            type = cms.string("l1t::TauPackerFactory"),
            Taus = cms.InputTag("Layer2HW")
            )
        ),
    InputLabel = cms.InputTag("Layer2HW"),
    FedId = cms.int32(100),
    FWId = cms.uint32(1)
)
