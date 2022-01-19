import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.DiDispStaMuonMonitor_cfi import hltDiDispStaMuonMonitoring


hltDiDispStaMuonCosmicMonitoring = hltDiDispStaMuonMonitoring.clone(
    FolderName = 'HLT/EXO/DiDispStaMuon/DoubleL2Mu23NoVtx_2Cha_CosmicSeed/'
)
hltDiDispStaMuonCosmicMonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleL2Mu23NoVtx_2Cha_CosmicSeed_v*") #HLT_ZeroBias_v*
## Efficiency trigger
hltDispStaMuon23Monitoring = hltDiDispStaMuonMonitoring.clone(
    FolderName = 'HLT/EXO/DiDispStaMuon/L2Mu23NoVtx_2Cha/'
)
hltDispStaMuon23Monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_L2Mu23NoVtx_2Cha_v*") #HLT_ZeroBias_v*

hltDispStaMuon23CosmicMonitoring = hltDiDispStaMuonMonitoring.clone(
    FolderName = 'HLT/EXO/DiDispStaMuon/L2Mu23NoVtx_2Cha_CosmicSeed/'
)
hltDispStaMuon23CosmicMonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_L2Mu23NoVtx_2Cha_CosmicSeed_v*") #HLT_ZeroBias_v*

## Backup trigger
hltDiDispStaMuon25Monitoring = hltDiDispStaMuonMonitoring.clone(
    FolderName = 'HLT/EXO/DiDispStaMuon/DoubleL2Mu25NoVtx_2Cha/'
)
hltDiDispStaMuon25Monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleL2Mu25NoVtx_2Cha_v*") #HLT_ZeroBias_v*

hltDiDispStaMuon25CosmicMonitoring = hltDiDispStaMuonMonitoring.clone(
    FolderName = 'HLT/EXO/DiDispStaMuon/DoubleL2Mu25NoVtx_2Cha_CosmicSeed/'
)
hltDiDispStaMuon25CosmicMonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleL2Mu25NoVtx_2Cha_CosmicSeed_v*") #HLT_ZeroBias_v*

hltDiDispStaMuon30Monitoring = hltDiDispStaMuonMonitoring.clone(
    FolderName = 'HLT/EXO/DiDispStaMuon/DoubleL2Mu30NoVtx_2Cha_Eta2p4/'
)
hltDiDispStaMuon30Monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleL2Mu30NoVtx_2Cha_Eta2p4_v*") #HLT_ZeroBias_v*

hltDiDispStaMuon30CosmicMonitoring = hltDiDispStaMuonMonitoring.clone(
    FolderName = 'HLT/EXO/DiDispStaMuon/DoubleL2Mu30NoVtx_2Cha_CosmicSeed_Eta2p4/'
)
hltDiDispStaMuon30CosmicMonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleL2Mu30NoVtx_2Cha_CosmicSeed_Eta2p4_v*") #HLT_ZeroBias_v*


exoHLTdispStaMuonMonitoring = cms.Sequence(
    hltDiDispStaMuonMonitoring
    + hltDiDispStaMuonCosmicMonitoring
    + hltDispStaMuon23Monitoring
    + hltDispStaMuon23CosmicMonitoring
    + hltDiDispStaMuon25Monitoring
    + hltDiDispStaMuon25CosmicMonitoring
    + hltDiDispStaMuon30Monitoring
    + hltDiDispStaMuon30CosmicMonitoring
)
