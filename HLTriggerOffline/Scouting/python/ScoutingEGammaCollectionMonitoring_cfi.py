import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

DoubleEGL1 = [
  "L1_DoubleEG_LooseIso16_LooseIso12_er1p5",
  "L1_DoubleEG_LooseIso18_LooseIso12_er1p5",
  "L1_DoubleEG_LooseIso20_LooseIso12_er1p5",
  "L1_DoubleEG_LooseIso22_LooseIso12_er1p5",
  "L1_DoubleEG11_er1p2_dR_Max0p6"
]

SinglePhotonL1 = [
  'L1_SingleLooseIsoEG26er2p5', 
  'L1_SingleLooseIsoEG26er1p5',
  'L1_SingleLooseIsoEG28er2p5',
  'L1_SingleLooseIsoEG28er2p1',
  'L1_SingleLooseIsoEG28er1p5',
  'L1_SingleLooseIsoEG30er2p5',
  'L1_SingleLooseIsoEG30er1p5',
  'L1_SingleEG26er2p5',
  'L1_SingleEG38er2p5',
  'L1_SingleEG40er2p5',
  'L1_SingleEG42er2p5',
  'L1_SingleEG45er2p5',
  'L1_SingleEG60', 
  'L1_SingleEG34er2p5',
  'L1_SingleEG36er2p5',
  'L1_SingleIsoEG24er2p1',
  'L1_SingleIsoEG26er2p1',
  'L1_SingleIsoEG28er2p1',
  'L1_SingleIsoEG30er2p1',
  'L1_SingleIsoEG32er2p1',
  'L1_SingleIsoEG26er2p5',
  'L1_SingleIsoEG28er2p5',
  'L1_SingleIsoEG30er2p5',
  'L1_SingleIsoEG32er2p5',
  'L1_SingleIsoEG34er2p5'
]

ScoutingEGammaCollectionMonitoring = DQMEDAnalyzer('ScoutingEGammaCollectionMonitoring',
                                                   OutputInternalPath = cms.string('/HLT/ScoutingOffline/EGamma/Collection'),
                                                   TriggerResultTag   = cms.InputTag("TriggerResults", "", "HLT"),
                                                   AlgInputTag  = cms.InputTag("gtStage2Digis"),
                                                   l1tAlgBlkInputTag  = cms.InputTag("gtStage2Digis"),
                                                   l1tExtBlkInputTag  = cms.InputTag("gtStage2Digis"),
                                                   ReadPrescalesFromFile = cms.bool(False),
                                                   triggerSelection   = cms.vstring(["DST_PFScouting_ZeroBias_v", "DST_PFScouting_DoubleEG_v", "DST_PFScouting_SinglePhotonEB_v"]),
                                                   L1Seeds            = cms.vstring(DoubleEGL1 + SinglePhotonL1),
                                                   ElectronCollection = cms.InputTag('slimmedElectrons'),
                                                   ScoutingElectronCollection = cms.InputTag("hltScoutingEgammaPacker"),
                                                   eleIdMapTight = cms.InputTag('egmGsfElectronIDsForScoutingDQM:cutBasedElectronID-RunIIIWinter22-V1-loose')
                                                   )

scoutingMonitoringEGM = cms.Sequence(ScoutingEGammaCollectionMonitoring)
