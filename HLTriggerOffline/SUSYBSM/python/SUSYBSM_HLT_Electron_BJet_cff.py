import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

SUSY_HLT_Electron_BJet = cms.EDAnalyzer("SUSY_HLT_Electron_BJet",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'), #to use with test sample
  ElectronCollection = cms.InputTag("gedGsfElectrons"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'), #to use with test sample
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_Ele10_CaloIdM_TrackIdM_CentralPFJet30_BTagCSV_p13_v'),
  TriggerFilterEle = cms.InputTag('hltSingleEle10CaloIdMTrackIdMDphiFilter', '', 'HLT'), #the last filter in the path hltSingleEle10CaloIdTrackIdVLOneOEMinusOneOPFilterRegional
  TriggerFilterJet = cms.InputTag('hltBTagPFCSVp13Single', '', 'HLT'), #the last filter in the path
  PtThrJet = cms.untracked.double(30.0),
  EtaThrJet = cms.untracked.double(3.0)
)

SUSY_HLT_Electron_BJet_POSTPROCESSING = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_Ele10_CaloIdM_TrackIdM_CentralPFJet30_BTagCSV_p13_v"),
    verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution     = cms.vstring(""),
    efficiency     = cms.vstring(
       "pfHTTurnOn_eff 'Turn-on vs HT; PFHT (GeV); #epsilon' pfHTTurnOn_num pfHTTurnOn_den",
       "EleTurnOn_eff 'Turn-on vs Mu pT; pT (GeV); #epsilon' EleTurnOn_num EleTurnOn_den",
    )
)
