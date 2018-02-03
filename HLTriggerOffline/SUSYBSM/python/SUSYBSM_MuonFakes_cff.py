import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
SUSY_HLT_Mu8_TrkIsoVVL = DQMEDAnalyzer('SUSY_HLT_MuonFakes',
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_Mu8_TrkIsoVVL_v'),
  TriggerFilter = cms.InputTag('hltL3fL1sMu5L1f0L2f5L3Filtered8TkIsoFiltered0p4', '', 'HLT'), #the last filter in the path                                        
)

SUSYoHLToMu8oTrkIsoVVLoPOSTPROCESSING = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_Mu8_TrkIsoVVL_v"),
    verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution     = cms.vstring(""),
    efficiency     = cms.vstring()
)

SUSY_HLT_Mu8 = DQMEDAnalyzer('SUSY_HLT_MuonFakes',
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_Mu8_v'),
  TriggerFilter = cms.InputTag('hltL3fL1sMu5L1f0L2f5L3Filtered8', '', 'HLT'), #the last filter in the path                                        
)

SUSYoHLToMu8oPOSTPROCESSING = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_Mu8_v"),
    verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution     = cms.vstring(""),
    efficiency     = cms.vstring()
)

SUSY_HLT_Mu17_TrkIsoVVL = DQMEDAnalyzer('SUSY_HLT_MuonFakes',
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_Mu17_TrkIsoVVL_v'),
  TriggerFilter = cms.InputTag('hltL3fL1sMu1lqL1f0L2f10L3Filtered17TkIsoFiltered0p4', '', 'HLT'), #the last filter in the path                                 
)

SUSYoHLToMu17oTrkIsoVVLoPOSTPROCESSING = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_Mu17_TrkIsoVVL_v"),
    verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution     = cms.vstring(""),
    efficiency     = cms.vstring()
)

SUSY_HLT_Mu17 = DQMEDAnalyzer('SUSY_HLT_MuonFakes',
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_Mu17_v'),
  TriggerFilter = cms.InputTag('hltL3fL1sMu10lqL1f0L2f10L3Filtered17', '', 'HLT'), #the last filter in the path                                 
)

SUSYoHLToMu17oPOSTPROCESSING = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_Mu17_v"),
    verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution     = cms.vstring(""),
    efficiency     = cms.vstring()
)

SUSY_HLT_TkMu17 = DQMEDAnalyzer('SUSY_HLT_MuonFakes',
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_TkMu17_v'),
  TriggerFilter = cms.InputTag('hltL3fL1sMu10lqTkFiltered17Q', '', 'HLT'), #the last filter in the path                                 
)

SUSYoHLToTkMu17oPOSTPROCESSING = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_TkMu17_v"),
    verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution     = cms.vstring(""),
    efficiency     = cms.vstring()
)


SUSY_HLT_MuonFakes = cms.Sequence(SUSY_HLT_Mu8 +
                                  SUSY_HLT_Mu17+
                                  SUSY_HLT_TkMu17+
                                  SUSY_HLT_Mu8_TrkIsoVVL+
                                  SUSY_HLT_Mu17_TrkIsoVVL)

SUSY_HLT_MuonFakes_POSTPROCESSING = cms.Sequence(SUSYoHLToMu8oPOSTPROCESSING +
                                                 SUSYoHLToMu17oPOSTPROCESSING+
                                                 SUSYoHLToTkMu17oPOSTPROCESSING+
                                                 SUSYoHLToMu8oTrkIsoVVLoPOSTPROCESSING+ 
                                                 SUSYoHLToMu17oTrkIsoVVLoPOSTPROCESSING)
