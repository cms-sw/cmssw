import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.NoBPTXMonitor_cfi import hltNoBPTXmonitoring

hltNoBPTXJetE70Monitoring = hltNoBPTXmonitoring.clone()
hltNoBPTXJetE70Monitoring.FolderName = cms.string('HLT/EXO/NoBPTX/JetE70/')
hltNoBPTXJetE70Monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_UncorrectedJetE70_NoBPTX3BX_v*") #HLT_ZeroBias_v*

hltNoBPTXL2Mu40Monitoring = hltNoBPTXmonitoring.clone()
hltNoBPTXL2Mu40Monitoring.FolderName = cms.string('HLT/EXO/NoBPTX/L2Mu40/')
hltNoBPTXL2Mu40Monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_L2Mu40_NoVertex_3Sta_NoBPTX3BX_v*") #HLT_ZeroBias_v*

hltNoBPTXL2Mu45Monitoring = hltNoBPTXmonitoring.clone()
hltNoBPTXL2Mu45Monitoring.FolderName = cms.string('HLT/EXO/NoBPTX/L2Mu45/')
hltNoBPTXL2Mu45Monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_L2Mu45_NoVertex_3Sta_NoBPTX3BX_v*") #HLT_ZeroBias_v*

exoHLTNoBPTXmonitoring = cms.Sequence(
    hltNoBPTXmonitoring
    + hltNoBPTXJetE70Monitoring
    + hltNoBPTXL2Mu40Monitoring
    + hltNoBPTXL2Mu45Monitoring
)
