import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
SUSY_HLT_DiJet_MET = DQMEDAnalyzer('SUSY_HLT_DiJet_MET',
  trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'), 
  pfMETCollection = cms.InputTag("pfMet"),
  caloMETCollection = cms.InputTag("caloMet"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  caloJetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_DiCentralPFJet55_PFMET110_v'),
  TriggerPathAuxiliaryForHadronic = cms.string('HLT_IsoMu24_eta2p1_v'),
  TriggerFilter = cms.InputTag('hltPFMET110Filter','','HLT'), #the last filter in the path
  TriggerJetFilter = cms.InputTag('hltDiCentralPFJet55','','HLT'), #the last filter in the path
  PtThrJetTrig = cms.untracked.double(55.0),
  EtaThrJetTrig = cms.untracked.double(2.6),
  PtThrJet = cms.untracked.double(30.0),
  EtaThrJet = cms.untracked.double(2.4),
  OfflineMetCut = cms.untracked.double(250.0),
)

SUSYoHLToDiJetMEToPOSTPROCESSING = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_DiCentralPFJet55_PFMET110_v"),
    verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution     = cms.vstring(""),
    efficiency     = cms.vstring(
       "pfMetTurnOn_eff 'Turn-on vs MET; PFMET (GeV); #epsilon' pfMetTurnOn_num pfMetTurnOn_den",
       "pfJet2PtTurnOn_eff 'Efficiency vs Jet2 p_{T}, NCentralPFJets >= 2, PFMET > 250 GeV; Second leading jet pT (GeV); #epsilon' pfJet2PtTurnOn_num pfJet2PtTurnOn_den",
    )
)
