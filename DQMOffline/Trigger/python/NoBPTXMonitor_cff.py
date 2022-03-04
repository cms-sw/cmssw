import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.NoBPTXMonitor_cfi import hltNoBPTXmonitoring

hltNoBPTXJetE70Monitoring = hltNoBPTXmonitoring.clone(
    FolderName = 'HLT/EXO/NoBPTX/JetE70/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_UncorrectedJetE70_NoBPTX3BX_v*"]) #HLT_ZeroBias_v*
)

hltNoBPTXL2Mu40Monitoring = hltNoBPTXmonitoring.clone(
    FolderName = 'HLT/EXO/NoBPTX/L2Mu40/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_L2Mu40_NoVertex_3Sta_NoBPTX3BX_v*"]) #HLT_ZeroBias_v*
)

hltNoBPTXL2Mu45Monitoring = hltNoBPTXmonitoring.clone(
    FolderName = 'HLT/EXO/NoBPTX/L2Mu45/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_L2Mu45_NoVertex_3Sta_NoBPTX3BX_v*"]) #HLT_ZeroBias_v*
)

exoHLTNoBPTXmonitoring = cms.Sequence(
    hltNoBPTXmonitoring
    + hltNoBPTXJetE70Monitoring
    + hltNoBPTXL2Mu40Monitoring
    + hltNoBPTXL2Mu45Monitoring
)
