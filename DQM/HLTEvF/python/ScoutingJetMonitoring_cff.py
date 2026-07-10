import FWCore.ParameterSet.Config as cms

## See DQMOffline/HLTScouting/python/HLTScoutingDqmOffline_cff.py

from DQMOffline.JetMET.jetMETDQMOfflineSource_cff import *
from DQMOffline.Trigger.JetMETPromptMonitor_cff import *

jetDQMOnlineAnalyzerAk4ScoutingCleaned = jetDQMAnalyzerAk4ScoutingCleaned.clone(
    JetType='scoutingOnline',
    DCSFilterForJetMonitoring=dict(DetectorTypes = "ecal:hbhe:hf:pixel:sistrip:es:muon",
                                   onlineMetaDataDigisSrc = cms.untracked.InputTag("hltOnlineMetaDataDigis"),
                                   DebugOn = cms.untracked.bool(False),
                                   alwaysPass = False)
)

jetDQMOnlineAnalyzerAk4ScoutingUncleaned = jetDQMAnalyzerAk4ScoutingUncleaned.clone(
    JetType='scoutingOnline',
    DCSFilterForJetMonitoring=dict(DetectorTypes = "ecal:hbhe:hf:pixel:sistrip:es:muon",
                                   onlineMetaDataDigisSrc = cms.untracked.InputTag("hltOnlineMetaDataDigis"),
                                   DebugOn =  cms.untracked.bool(False),
                                   alwaysPass = False)
)
dqmAk4PFScoutingL1FastL2L3ResidualCorrectorChain = cms.Sequence(dqmAk4PFScoutingL1FastL2L3ResidualCorrectorTask)
jetDQMOnlineAnalyzerSequenceScouting = cms.Sequence(jetDQMOnlineAnalyzerAk4ScoutingUncleaned*
                                                    jetDQMOnlineAnalyzerAk4ScoutingCleaned)


######### Scouitng Trigger and L1 seeds #########
#PFScoutingJetHT
PFScoutingJetHT_Onlinemonitoring = PFScoutingJetHT_Prommonitoring.clone(
    FolderName = 'HLT/ScoutingOnline/Jet/PFScoutingJetHT/'
)

# L1_HTT200er
L1HTT200_Onlinemonitoring = L1HTT200_Prommonitoring.clone(
    FolderName = 'HLT/ScoutingOnline/Jet/L1_HTT200er/'
)


# L1_HTT255er
L1HTT255_Onlinemonitoring = L1HTT255_Prommonitoring.clone(
    FolderName = 'HLT/ScoutingOnline/Jet/L1_HTT255er/'
)


# L1_HTT280er
L1HTT280_Onlinemonitoring = L1HTT280_Prommonitoring.clone(
    FolderName = 'HLT/ScoutingOnline/Jet/L1_HTT280er/'
)


#  L1_HTT320er
L1HTT320_Onlinemonitoring = L1HTT320_Prommonitoring.clone(
    FolderName = 'HLT/ScoutingOnline/Jet/L1_HTT320er/'
)


# L1_HTT360er
L1HTT360_Onlinemonitoring = L1HTT360_Prommonitoring.clone(
    FolderName = 'HLT/ScoutingOnline/Jet/L1_HTT360er/'
)

# L1_HTT400er
L1HTT400_Onlinemonitoring = L1HTT400_Prommonitoring.clone(
    FolderName = 'HLT/ScoutingOnline/Jet/L1_HTT400er/'
)


# L1_HTT450er
L1HTT450_Onlinemonitoring = L1HTT450_Prommonitoring.clone(
    FolderName = 'HLT/ScoutingOnline/Jet/L1_HTT450er/'
)


# L1_SingleJet180
L1SingleJet180_Onlinemonitoring = L1SingleJet180_Prommonitoring.clone(
    FolderName = 'HLT/ScoutingOnline/Jet/L1_SingleJet180/'
)


# L1_SingleJet200
L1SingleJet200_Onlinemonitoring = L1SingleJet200_Prommonitoring.clone(
    FolderName = 'HLT/ScoutingOnline/Jet/L1_SingleJet200/'
)

HLTScoutingJetOnlinemonitoring = cms.Sequence(
    ak4PFScoutL1FastL2L3ResidualCorrectorChain
    *PFScoutingJetHT_Onlinemonitoring
    *L1HTT200_Onlinemonitoring
    *L1HTT255_Onlinemonitoring
    *L1HTT280_Onlinemonitoring
    *L1HTT320_Onlinemonitoring
    *L1HTT360_Onlinemonitoring
    *L1HTT400_Onlinemonitoring
    *L1HTT450_Onlinemonitoring
    *L1SingleJet180_Onlinemonitoring
    *L1SingleJet200_Onlinemonitoring
)

jetmetScoutingOnlineMonitorHLT = cms.Sequence(
    HLTScoutingJetOnlinemonitoring
)


ScoutingJetMonitoring = cms.Sequence(jetPreDQMSeqScouting*
                                     dqmAk4PFScoutingL1FastL2L3ResidualCorrectorChain*
                                     jetDQMOnlineAnalyzerSequenceScouting*
                                     jetmetScoutingOnlineMonitorHLT)
