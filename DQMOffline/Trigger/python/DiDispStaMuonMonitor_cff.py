import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.DiDispStaMuonMonitor_cfi import hltDiDispStaMuonMonitoring


hltDiDispStaMuonCosmicMonitoring = hltDiDispStaMuonMonitoring.clone(
    FolderName = 'HLT/EXO/DiDispStaMuon/DoubleL2Mu23NoVtx_2Cha_CosmicSeed/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_DoubleL2Mu23NoVtx_2Cha_CosmicSeed_v*"]) #HLT_ZeroBias_v*
)

hltDiDispStaMuon10VetoL3Mu0DxyMax1cmMonitoring = hltDiDispStaMuonMonitoring.clone(
    FolderName = 'HLT/EXO/DiDispStaMuon/DoubleL2Mu10NoVtx_2Cha_VetoL3Mu0DxyMax1cm/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_DoubleL2Mu10NoVtx_2Cha_VetoL3Mu0DxyMax1cm_v*"]) #HLT_ZeroBias_v*
)

hltDiDispStaMuonL2MuL3Mu16VetoL3Mu0DxyMax1cmMonitoring = hltDiDispStaMuonMonitoring.clone(
    FolderName = 'HLT/EXO/DiDispStaMuon/DoubleL2Mu_L3Mu16NoVtx_VetoL3Mu0DxyMax0p1cm/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_DoubleL2Mu_L3Mu16NoVtx_VetoL3Mu0DxyMax0p1cm_v*"]) #HLT_ZeroBias_v*
)

hltDiDispStaMuon10CosmicVetoL3Mu0DxyMax1cmMonitoring = hltDiDispStaMuonMonitoring.clone(
    FolderName = 'HLT/EXO/DiDispStaMuon/DoubleL2Mu10NoVtx_2Cha_CosmicSeed_VetoL3Mu0DxyMax1cm/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_DoubleL2Mu10NoVtx_2Cha_CosmicSeed_VetoL3Mu0DxyMax1cm_v*"]) #HLT_ZeroBias_v*
)

exoHLTdispStaMuonMonitoring = cms.Sequence(
    hltDiDispStaMuonMonitoring
    + hltDiDispStaMuonCosmicMonitoring
    + hltDiDispStaMuon10VetoL3Mu0DxyMax1cmMonitoring
    + hltDiDispStaMuonL2MuL3Mu16VetoL3Mu0DxyMax1cmMonitoring
    + hltDiDispStaMuon10CosmicVetoL3Mu0DxyMax1cmMonitoring
)
