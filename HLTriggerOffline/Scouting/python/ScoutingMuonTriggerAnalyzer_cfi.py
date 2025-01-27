import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

DoubleMuL1 = ["L1_DoubleMu_15_7","L1_DoubleMu4p5er2p0_SQ_OS_Mass_Min7","L1_DoubleMu4p5er2p0_SQ_OS_Mass_7to18","L1_DoubleMu8_SQ","L1_DoubleMu4er2p0_SQ_OS_dR_Max1p6","L1_DoubleMu0er1p4_SQ_OS_dR_Max1p4","L1_DoubleMu4p5_SQ_OS_dR_Max1p2","L1_DoubleMu0_Upt15_Upt7","L1_DoubleMu0_Upt6_IP_Min1_Upt4"]
SingleMuL1 = ["L1_SingleMu11_SQ14_BMTF"]
#zeroBias = "DST_ZeroBias_v*"

ScoutingMuonTriggerAnalysis = DQMEDAnalyzer('ScoutingMuonTriggerAnalyzer',
    OutputInternalPath = cms.string('/HLT/ScoutingOffline/Muons/L1Efficiency'),
    MuonCollection = cms.InputTag('slimmedMuons'),
    ScoutingMuonCollection = cms.InputTag('hltScoutingMuonPackerVtx'),
    triggerSelection = cms.vstring(["DST_ZeroBias_v*", "DST_PFScouting_DoubleEG_v*", "DST_PFScouting_JetHT_v*"]), # Updated in 2024
    triggerConfiguration = cms.PSet(
        hltResults            = cms.InputTag('TriggerResults','','HLT'),
        l1tResults            = cms.InputTag('','',''),
        l1tIgnoreMaskAndPrescale = cms.bool(False),
        throw                 = cms.bool(True),
        usePathStatus = cms.bool(False),
    ),
    AlgInputTag = cms.InputTag("gtStage2Digis"),
    l1tAlgBlkInputTag = cms.InputTag("gtStage2Digis"),
    l1tExtBlkInputTag = cms.InputTag("gtStage2Digis"),
    ReadPrescalesFromFile = cms.bool(False),
    l1Seeds = cms.vstring(SingleMuL1+DoubleMuL1)
    #ScoutingTriggerCollection = cms.InputTag('TriggerResults'),
)

scoutingMonitoringTriggerMuon = cms.Sequence(ScoutingMuonTriggerAnalysis)