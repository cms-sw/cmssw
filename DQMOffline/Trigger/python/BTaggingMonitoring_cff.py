import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.BTaggingMonitor_cfi import hltBTVmonitoring

#BTagMu AK4
BTagMu_AK4DiJet20_Mu5 = hltBTVmonitoring.clone()
BTagMu_AK4DiJet20_Mu5.FolderName = cms.string('HLT/BTV/BTagMu_DiJet/BTagMu_AK4DiJet20_Mu5')
BTagMu_AK4DiJet20_Mu5.nmuons = cms.uint32(2)
BTagMu_AK4DiJet20_Mu5.nelectrons = cms.uint32(0)
BTagMu_AK4DiJet20_Mu5.njets = cms.uint32(2)
BTagMu_AK4DiJet20_Mu5.muoSelection = cms.string('pt>3 & abs(eta)<2.4')
BTagMu_AK4DiJet20_Mu5.jetSelection = cms.string('pt>10 & abs(eta)<2.4')
BTagMu_AK4DiJet20_Mu5.bjetSelection = cms.string('pt>5 & abs(eta)<2.4')
BTagMu_AK4DiJet20_Mu5.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_BTagMu_AK4DiJet20_Mu5_v*')

BTagMu_AK4DiJet40_Mu5 = hltBTVmonitoring.clone()
BTagMu_AK4DiJet40_Mu5.FolderName = cms.string('HLT/BTV/BTagMu_DiJet/BTagMu_AK4DiJet40_Mu5')
BTagMu_AK4DiJet40_Mu5.nmuons = cms.uint32(2)
BTagMu_AK4DiJet40_Mu5.nelectrons = cms.uint32(0)
BTagMu_AK4DiJet40_Mu5.njets = cms.uint32(2)
BTagMu_AK4DiJet40_Mu5.muoSelection = cms.string('pt>3 & abs(eta)<2.4')
BTagMu_AK4DiJet40_Mu5.jetSelection = cms.string('pt>30 & abs(eta)<2.4')
BTagMu_AK4DiJet40_Mu5.bjetSelection = cms.string('pt>20 & abs(eta)<2.4')
BTagMu_AK4DiJet40_Mu5.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_BTagMu_AK4DiJet40_Mu5_v*')

BTagMu_AK4DiJet70_Mu5 = hltBTVmonitoring.clone()
BTagMu_AK4DiJet70_Mu5.FolderName = cms.string('HLT/BTV/BTagMu_DiJet/BTagMu_AK4DiJet70_Mu5')
BTagMu_AK4DiJet70_Mu5.nmuons = cms.uint32(2)
BTagMu_AK4DiJet70_Mu5.nelectrons = cms.uint32(0)
BTagMu_AK4DiJet70_Mu5.njets = cms.uint32(2)
BTagMu_AK4DiJet70_Mu5.muoSelection = cms.string('pt>3 & abs(eta)<2.4')
BTagMu_AK4DiJet70_Mu5.jetSelection = cms.string('pt>50 & abs(eta)<2.4')
BTagMu_AK4DiJet70_Mu5.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_BTagMu_AK4DiJet70_Mu5_v*')

BTagMu_AK4DiJet110_Mu5 = hltBTVmonitoring.clone()
BTagMu_AK4DiJet110_Mu5.FolderName = cms.string('HLT/BTV/BTagMu_DiJet/BTagMu_AK4DiJet110_Mu5')
BTagMu_AK4DiJet110_Mu5.nmuons = cms.uint32(2)
BTagMu_AK4DiJet110_Mu5.nelectrons = cms.uint32(0)
BTagMu_AK4DiJet110_Mu5.njets = cms.uint32(2)
BTagMu_AK4DiJet110_Mu5.muoSelection = cms.string('pt>3 & abs(eta)<2.4')
BTagMu_AK4DiJet110_Mu5.jetSelection = cms.string('pt>90 & abs(eta)<2.4')
BTagMu_AK4DiJet110_Mu5.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_BTagMu_AK4DiJet110_Mu5_v*')

BTagMu_AK4DiJet170_Mu5 = hltBTVmonitoring.clone()
BTagMu_AK4DiJet170_Mu5.FolderName = cms.string('HLT/BTV/BTagMu_DiJet/BTagMu_AK4DiJet170_Mu5')
BTagMu_AK4DiJet170_Mu5.nmuons = cms.uint32(2)
BTagMu_AK4DiJet170_Mu5.nelectrons = cms.uint32(0)
BTagMu_AK4DiJet170_Mu5.njets = cms.uint32(2)
BTagMu_AK4DiJet170_Mu5.muoSelection = cms.string('pt>3 & abs(eta)<2.4')
BTagMu_AK4DiJet170_Mu5.jetSelection = cms.string('pt>150 & abs(eta)<2.4')
BTagMu_AK4DiJet170_Mu5.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_BTagMu_AK4DiJet170_Mu5_v*')

BTagMu_AK4Jet300_Mu5 = hltBTVmonitoring.clone()
BTagMu_AK4Jet300_Mu5.FolderName = cms.string('HLT/BTV/BTagMu_Jet/BTagMu_AK4Jet300_Mu5')
BTagMu_AK4Jet300_Mu5.nmuons = cms.uint32(1)
BTagMu_AK4Jet300_Mu5.nelectrons = cms.uint32(0)
BTagMu_AK4Jet300_Mu5.njets = cms.uint32(1)
BTagMu_AK4Jet300_Mu5.muoSelection = cms.string('pt>3 & abs(eta)<2.4')
BTagMu_AK4Jet300_Mu5.jetSelection = cms.string('pt>250 & abs(eta)<2.4')
BTagMu_AK4Jet300_Mu5.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_BTagMu_AK4Jet300_Mu5_v*')

#BTagMu AK8
BTagMu_AK8DiJet170_Mu5 = hltBTVmonitoring.clone()
BTagMu_AK8DiJet170_Mu5.FolderName = cms.string('HLT/BTV/BTagMu_DiJet/BTagMu_AK8DiJet170_Mu5')
BTagMu_AK8DiJet170_Mu5.nmuons = cms.uint32(2)
BTagMu_AK8DiJet170_Mu5.nelectrons = cms.uint32(0)
BTagMu_AK8DiJet170_Mu5.njets = cms.uint32(2)
BTagMu_AK8DiJet170_Mu5.jets = cms.InputTag("ak8PFJetsCHS")
BTagMu_AK8DiJet170_Mu5.muoSelection = cms.string('pt>3 & abs(eta)<2.4')
BTagMu_AK8DiJet170_Mu5.jetSelection = cms.string('pt>150 & abs(eta)<2.4')
BTagMu_AK8DiJet170_Mu5.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_BTagMu_AK8DiJet170_Mu5_v*')

BTagMu_AK8Jet300_Mu5 = hltBTVmonitoring.clone()
BTagMu_AK8Jet300_Mu5.FolderName = cms.string('HLT/BTV/BTagMu_Jet/BTagMu_AK8Jet300_Mu5')
BTagMu_AK8Jet300_Mu5.nmuons = cms.uint32(1)
BTagMu_AK8Jet300_Mu5.nelectrons = cms.uint32(0)
BTagMu_AK8Jet300_Mu5.njets = cms.uint32(1)
BTagMu_AK8Jet300_Mu5.jets = cms.InputTag("ak8PFJetsCHS")
BTagMu_AK8Jet300_Mu5.muoSelection = cms.string('pt>3 & abs(eta)<2.4')
BTagMu_AK8Jet300_Mu5.jetSelection = cms.string('pt>250 & abs(eta)<2.4')
BTagMu_AK8Jet300_Mu5.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_BTagMu_AK8Jet300_Mu5_v*')

#PFJet AK4
PFJet40 = hltBTVmonitoring.clone()
PFJet40.FolderName = cms.string('HLT/BTV/PFJet/PFJet40')
PFJet40.nmuons = cms.uint32(0)
PFJet40.nelectrons = cms.uint32(0)
PFJet40.njets = cms.uint32(1)
PFJet40.jetSelection = cms.string('pt>30 & abs(eta)<2.4')
PFJet40.bjetSelection = cms.string('pt>20 & abs(eta)<2.4')
PFJet40.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFJet40_v*')

PFJet60 = hltBTVmonitoring.clone()
PFJet60.FolderName = cms.string('HLT/BTV/PFJet/PFJet60')
PFJet60.nmuons = cms.uint32(0)
PFJet60.nelectrons = cms.uint32(0)
PFJet60.njets = cms.uint32(1)
PFJet60.jetSelection = cms.string('pt>50 & abs(eta)<2.4')
PFJet60.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFJet60_v*')

PFJet80 = hltBTVmonitoring.clone()
PFJet80.FolderName = cms.string('HLT/BTV/PFJet/PFJet80')
PFJet80.nmuons = cms.uint32(0)
PFJet80.nelectrons = cms.uint32(0)
PFJet80.njets = cms.uint32(1)
PFJet80.jetSelection = cms.string('pt>70 & abs(eta)<2.4')
PFJet80.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFJet80_v*')

PFJet140 = hltBTVmonitoring.clone()
PFJet140.FolderName = cms.string('HLT/BTV/PFJet/PFJet140')
PFJet140.nmuons = cms.uint32(0)
PFJet140.nelectrons = cms.uint32(0)
PFJet140.njets = cms.uint32(1)
PFJet140.jetSelection = cms.string('pt>120 & abs(eta)<2.4')
PFJet140.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFJet140_v*')

PFJet200 = hltBTVmonitoring.clone()
PFJet200.FolderName = cms.string('HLT/BTV/PFJet/PFJet200')
PFJet200.nmuons = cms.uint32(0)
PFJet200.nelectrons = cms.uint32(0)
PFJet200.njets = cms.uint32(1)
PFJet200.jetSelection = cms.string('pt>170 & abs(eta)<2.4')
PFJet200.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFJet200_v*')

PFJet260 = hltBTVmonitoring.clone()
PFJet260.FolderName = cms.string('HLT/BTV/PFJet/PFJet260')
PFJet260.nmuons = cms.uint32(0)
PFJet260.nelectrons = cms.uint32(0)
PFJet260.njets = cms.uint32(1)
PFJet260.jetSelection = cms.string('pt>220 & abs(eta)<2.4')
PFJet260.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFJet260_v*')

PFJet320 = hltBTVmonitoring.clone()
PFJet320.FolderName = cms.string('HLT/BTV/PFJet/PFJet320')
PFJet320.nmuons = cms.uint32(0)
PFJet320.nelectrons = cms.uint32(0)
PFJet320.njets = cms.uint32(1)
PFJet320.jetSelection = cms.string('pt>280 & abs(eta)<2.4')
PFJet320.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFJet320_v*')

PFJet400 = hltBTVmonitoring.clone()
PFJet400.FolderName = cms.string('HLT/BTV/PFJet/PFJet400')
PFJet400.nmuons = cms.uint32(0)
PFJet400.nelectrons = cms.uint32(0)
PFJet400.njets = cms.uint32(1)
PFJet400.jetSelection = cms.string('pt>350 & abs(eta)<2.4')
PFJet400.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFJet400_v*')

PFJet450 = hltBTVmonitoring.clone()
PFJet450.FolderName = cms.string('HLT/BTV/PFJet/PFJet450')
PFJet450.nmuons = cms.uint32(0)
PFJet450.nelectrons = cms.uint32(0)
PFJet450.njets = cms.uint32(1)
PFJet450.jetSelection = cms.string('pt>400 & abs(eta)<2.4')
PFJet450.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFJet450_v*')

PFJet500 = hltBTVmonitoring.clone()
PFJet500.FolderName = cms.string('HLT/BTV/PFJet/PFJet500')
PFJet500.nmuons = cms.uint32(0)
PFJet500.nelectrons = cms.uint32(0)
PFJet500.njets = cms.uint32(1)
PFJet500.jetSelection = cms.string('pt>450 & abs(eta)<2.4')
PFJet500.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_PFJet500_v*')

#PFJet AK8
AK8PFJet40 = hltBTVmonitoring.clone()
AK8PFJet40.FolderName = cms.string('HLT/BTV/PFJet/AK8PFJet40')
AK8PFJet40.nmuons = cms.uint32(0)
AK8PFJet40.nelectrons = cms.uint32(0)
AK8PFJet40.njets = cms.uint32(1)
AK8PFJet40.jets = cms.InputTag("ak8PFJetsCHS")
AK8PFJet40.jetSelection = cms.string('pt>30 & abs(eta)<2.4')
AK8PFJet40.bjetSelection = cms.string('pt>20 & abs(eta)<2.4')
AK8PFJet40.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_AK8PFJet40_v*')

AK8PFJet60 = hltBTVmonitoring.clone()
AK8PFJet60.FolderName = cms.string('HLT/BTV/PFJet/AK8PFJet60')
AK8PFJet60.nmuons = cms.uint32(0)
AK8PFJet60.nelectrons = cms.uint32(0)
AK8PFJet60.njets = cms.uint32(1)
AK8PFJet60.jets = cms.InputTag("ak8PFJetsCHS")
AK8PFJet60.jetSelection = cms.string('pt>50 & abs(eta)<2.4')
AK8PFJet60.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_AK8PFJet60_v*')

AK8PFJet80 = hltBTVmonitoring.clone()
AK8PFJet80.FolderName = cms.string('HLT/BTV/PFJet/AK8PFJet80')
AK8PFJet80.nmuons = cms.uint32(0)
AK8PFJet80.nelectrons = cms.uint32(0)
AK8PFJet80.njets = cms.uint32(1)
AK8PFJet80.jets = cms.InputTag("ak8PFJetsCHS")
AK8PFJet80.jetSelection = cms.string('pt>70 & abs(eta)<2.4')
AK8PFJet80.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_AK8PFJet80_v*')

AK8PFJet140 = hltBTVmonitoring.clone()
AK8PFJet140.FolderName = cms.string('HLT/BTV/PFJet/AK8PFJet140')
AK8PFJet140.nmuons = cms.uint32(0)
AK8PFJet140.nelectrons = cms.uint32(0)
AK8PFJet140.njets = cms.uint32(1)
AK8PFJet140.jets = cms.InputTag("ak8PFJetsCHS")
AK8PFJet140.jetSelection = cms.string('pt>120 & abs(eta)<2.4')
AK8PFJet140.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_AK8PFJet140_v*')

AK8PFJet200 = hltBTVmonitoring.clone()
AK8PFJet200.FolderName = cms.string('HLT/BTV/PFJet/AK8PFJet200')
AK8PFJet200.nmuons = cms.uint32(0)
AK8PFJet200.nelectrons = cms.uint32(0)
AK8PFJet200.njets = cms.uint32(1)
AK8PFJet200.jets = cms.InputTag("ak8PFJetsCHS")
AK8PFJet200.jetSelection = cms.string('pt>170 & abs(eta)<2.4')
AK8PFJet200.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_AK8PFJet200_v*')

AK8PFJet260 = hltBTVmonitoring.clone()
AK8PFJet260.FolderName = cms.string('HLT/BTV/PFJet/AK8PFJet260')
AK8PFJet260.nmuons = cms.uint32(0)
AK8PFJet260.nelectrons = cms.uint32(0)
AK8PFJet260.njets = cms.uint32(1)
AK8PFJet260.jets = cms.InputTag("ak8PFJetsCHS")
AK8PFJet260.jetSelection = cms.string('pt>220 & abs(eta)<2.4')
AK8PFJet260.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_AK8PFJet260_v*')

AK8PFJet320 = hltBTVmonitoring.clone()
AK8PFJet320.FolderName = cms.string('HLT/BTV/PFJet/AK8PFJet320')
AK8PFJet320.nmuons = cms.uint32(0)
AK8PFJet320.nelectrons = cms.uint32(0)
AK8PFJet320.njets = cms.uint32(1)
AK8PFJet320.jets = cms.InputTag("ak8PFJetsCHS")
AK8PFJet320.jetSelection = cms.string('pt>280 & abs(eta)<2.4')
AK8PFJet320.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_AK8PFJet320_v*')

AK8PFJet400 = hltBTVmonitoring.clone()
AK8PFJet400.FolderName = cms.string('HLT/BTV/PFJet/AK8PFJet400')
AK8PFJet400.nmuons = cms.uint32(0)
AK8PFJet400.nelectrons = cms.uint32(0)
AK8PFJet400.njets = cms.uint32(1)
AK8PFJet400.jets = cms.InputTag("ak8PFJetsCHS")
AK8PFJet400.jetSelection = cms.string('pt>350 & abs(eta)<2.4')
AK8PFJet400.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_AK8PFJet400_v*')

AK8PFJet450 = hltBTVmonitoring.clone()
AK8PFJet450.FolderName = cms.string('HLT/BTV/PFJet/AK8PFJet450')
AK8PFJet450.nmuons = cms.uint32(0)
AK8PFJet450.nelectrons = cms.uint32(0)
AK8PFJet450.njets = cms.uint32(1)
AK8PFJet450.jets = cms.InputTag("ak8PFJetsCHS")
AK8PFJet450.jetSelection = cms.string('pt>400 & abs(eta)<2.4')
AK8PFJet450.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_AK8PFJet450_v*')

AK8PFJet500 = hltBTVmonitoring.clone()
AK8PFJet500.FolderName = cms.string('HLT/BTV/PFJet/AK8PFJet500')
AK8PFJet500.nmuons = cms.uint32(0)
AK8PFJet500.nelectrons = cms.uint32(0)
AK8PFJet500.njets = cms.uint32(1)
AK8PFJet500.jets = cms.InputTag("ak8PFJetsCHS")
AK8PFJet500.jetSelection = cms.string('pt>450 & abs(eta)<2.4')
AK8PFJet500.numGenericTriggerEventPSet.hltPaths = cms.vstring('HLT_AK8PFJet500_v*')


btagMonitorHLT = cms.Sequence(
    BTagMu_AK4DiJet20_Mu5
    + BTagMu_AK4DiJet40_Mu5
    + BTagMu_AK4DiJet70_Mu5
    + BTagMu_AK4DiJet110_Mu5    
    + BTagMu_AK4DiJet170_Mu5
    + BTagMu_AK4Jet300_Mu5
    + BTagMu_AK8DiJet170_Mu5
    + BTagMu_AK8Jet300_Mu5
    + PFJet40
    + PFJet60
    + PFJet80
    + PFJet140
    + PFJet200
    + PFJet260
    + PFJet320
    + PFJet400
    + PFJet450
    + PFJet500
    + AK8PFJet40
    + AK8PFJet60
    + AK8PFJet80
    + AK8PFJet140
    + AK8PFJet200
    + AK8PFJet260
    + AK8PFJet320
    + AK8PFJet400
    + AK8PFJet450
    + AK8PFJet500
)
