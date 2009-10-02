import FWCore.ParameterSet.Config as cms

higgsToWW2LeptonsFakeRatesFilter = cms.EDFilter("HiggsToWW2LeptonsSkim",
    ElectronCollectionLabel = cms.InputTag("gsfElectrons"),
    SingleLeptonPtMin = cms.double(10.0),
    nLeptons=cms.int32(1),
    etaMin = cms.double(-2.4),
    MuonCollectionLabel = cms.InputTag("muons"),
    muonType=cms.string('All'),
    DiLeptonPtMin = cms.double(0.0),
    etaMax = cms.double(2.4),
    beTight = cms.bool(False),
    dilepM = cms.double(6),
    eleHadronicOverEm = cms.double(0.5)

)


