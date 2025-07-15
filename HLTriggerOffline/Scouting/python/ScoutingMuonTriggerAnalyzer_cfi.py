'''
This code specifies which Double Muon and Single Muon triggers (numerator) and 
HLTriggers, as defined in triggerSelection (denominator) to use in  
ScoutingMuonTriggerAnalyzer.cc, and what cuts to apply in both SingleMu and 
DoubleMu triggers.

Author: Javier Garcia de Castro, email:javigdc@bu.edu
'''

import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

#List of Double and Single Muon triggers (numerator)
DoubleMuL1 = [
    "L1_DoubleMu0_Upt8_SQ_er2p0",
    "L1_DoubleMu0_Upt7_SQ_er2p0",
    "L1_DoubleMu_15_7",
    "L1_DoubleMu4p5er2p0_SQ_OS_Mass_Min7",
    "L1_DoubleMu4p5er2p0_SQ_OS_Mass_7to18",
    "L1_DoubleMu8_SQ",
    "L1_DoubleMu4er2p0_SQ_OS_dR_Max1p6",
    "L1_DoubleMu0er1p4_SQ_OS_dR_Max1p4",
    "L1_DoubleMu4p5_SQ_OS_dR_Max1p2",
    "L1_DoubleMu0_Upt15_Upt7",
    "L1_DoubleMu0_Upt6_IP_Min1_Upt4",
    "L1_DoubleMu0_Upt6_SQ_er2p0"
]
SingleMuL1 = ["L1_SingleMu11_SQ14_BMTF","L1_SingleMu10_SQ14_BMTF"]

ScoutingMuonTriggerAnalysis_DoubleMu = DQMEDAnalyzer('ScoutingMuonTriggerAnalyzer',
    OutputInternalPath = cms.string('/HLT/ScoutingOffline/Muons/L1Efficiency/DoubleMu'), #Output of the root file
    ScoutingMuonCollection = cms.InputTag('hltScoutingMuonPackerVtx'),
    triggerSelection = cms.vstring(["DST_PFScouting_ZeroBias*", "DST_PFScouting_DoubleEG_v*", "DST_PFScouting_JetHT_v*"]), #Denominator
    triggerConfiguration = cms.PSet(
        hltResults            = cms.InputTag('TriggerResults','','HLT'),
        l1tResults            = cms.InputTag('','',''),
        l1tIgnoreMaskAndPrescale = cms.bool(False),
        throw                 = cms.bool(False),
        usePathStatus = cms.bool(False),
    ),
    AlgInputTag = cms.InputTag("gtStage2Digis"),
    l1tAlgBlkInputTag = cms.InputTag("gtStage2Digis"),
    l1tExtBlkInputTag = cms.InputTag("gtStage2Digis"),
    ReadPrescalesFromFile = cms.bool(False),
    l1Seeds = cms.vstring(DoubleMuL1), #Full list of double muon L1 seeds
    muonSelection = cms.string("")   #No cuts applied to double muon triggers
)

ScoutingMuonTriggerAnalysis_SingleMu = ScoutingMuonTriggerAnalysis_DoubleMu.clone(
    OutputInternalPath = cms.string('/HLT/ScoutingOffline/Muons/L1Efficiency/SingleMu'),
    l1Seeds = cms.vstring(SingleMuL1),  #Full list of single muon L1 seeds
    muonSelection = cms.string("abs(eta)<0.8") #Eta cut applied to single muons
)
#Name given to add to the sequence in test/runScoutingMonitoringDQM_muonOnly_cfg.py
scoutingMonitoringTriggerMuon_DoubleMu = cms.Sequence(ScoutingMuonTriggerAnalysis_DoubleMu)
scoutingMonitoringTriggerMuon_SingleMu = cms.Sequence(ScoutingMuonTriggerAnalysis_SingleMu)
