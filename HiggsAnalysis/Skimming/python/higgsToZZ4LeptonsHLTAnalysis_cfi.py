import FWCore.ParameterSet.Config as cms

higgsToZZ4LeptonsHLTAnalysis = cms.EDProducer("HiggsToZZ4LeptonsHLTAnalysis",
    ElectronCollectionLabel = cms.InputTag("pixelMatchGsfElectrons"),
##Summer08
HLTPaths = cms.vstring('HLT_LooseIsoEle15_LW_L1R', 'HLT_DoubleEle10_LW_OnlyPixelM_L1R', 'HLT_IsoMu9', 'HLT_DoubleMu3'),
##STARTUP 31X
#HLTPaths = cms.vstring('HLT_Ele10_LW_EleId_L1R', 'HLT_DoubleEle5_SW__L1R', 'HLT_Mu9', 'HLT_DoubleMu3'),
##IDEAL 31X  
#HLTPaths = cms.vstring('HLT_Ele15_SW_LooseTrackIso_L1R', 'HLT_DoubleEle10_SW_L1R', 'HLT_IsoMu9', 'HLT_DoubleMu3'),
    andOr = cms.bool(True),
    MuonCollectionLabel = cms.InputTag("muons"),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)


