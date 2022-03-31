import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.NoBPTXMonitor_cfi import hltNoBPTXmonitoring

hltNoBPTXL2Mu40Monitoring = hltNoBPTXmonitoring.clone(
    FolderName = 'HLT/EXO/NoBPTX/L2Mu40/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_L2Mu40_NoVertex_3Sta_NoBPTX3BX_v*"]) #HLT_ZeroBias_v*
)

exoHLTNoBPTXmonitoring = cms.Sequence(
    hltNoBPTXmonitoring
    + hltNoBPTXL2Mu40Monitoring
)
