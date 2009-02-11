import FWCore.ParameterSet.Config as cms

higgsToZZ4LeptonsHLTAnalysis = cms.EDProducer("HiggsToZZ4LeptonsHLTAnalysis",
    ElectronCollectionLabel = cms.InputTag("pixelMatchGsfElectrons"),
    HLTPaths = cms.vstring('HLT_IsoMu11', 'HLT_DoubleMu3', 'HLT_IsoEle15_L1I', 'HLT_IsoEle18_L1R', 'HLT_DoubleIsoEle10_L1I', 'HLT_DoubleIsoEle12_L1R'),
    andOr = cms.bool(True),
    MuonCollectionLabel = cms.InputTag("muons"),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLTreprocess")
)


