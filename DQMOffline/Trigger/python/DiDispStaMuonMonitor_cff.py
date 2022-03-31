import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.DiDispStaMuonMonitor_cfi import hltDiDispStaMuonMonitoring


hltDiDispStaMuonCosmicMonitoring = hltDiDispStaMuonMonitoring.clone(
    FolderName = 'HLT/EXO/DiDispStaMuon/DoubleL2Mu23NoVtx_2Cha_CosmicSeed/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_DoubleL2Mu23NoVtx_2Cha_CosmicSeed_v*"]) #HLT_ZeroBias_v*
)

hltDiDispStaMuon10PromptL3Mu0VetoMonitoring = hltDiDispStaMuonMonitoring.clone(
    FolderName = 'HLT/EXO/DiDispStaMuon/DoubleL2Mu10NoVtx_2Cha_PromptL3Mu0Veto/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_DoubleL2Mu10NoVtx_2Cha_PromptL3Mu0Veto_v*"]) #HLT_ZeroBias_v*
)

exoHLTdispStaMuonMonitoring = cms.Sequence(
    hltDiDispStaMuonMonitoring
    + hltDiDispStaMuonCosmicMonitoring
    + hltDiDispStaMuon10PromptL3Mu0VetoMonitoring
)
