import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

SUSY_HLT_Mu8_TrkIsoVVL = cms.EDAnalyzer("SUSY_HLT_MuonFakes",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_Mu8_TrkIsoVVL_v'),
  TriggerFilter = cms.InputTag('hltL3fL1sMu5L1f0L2f5L3Filtered8TkIsoFiltered0p4', '', 'HLT'), #the last filter in the path                                        
)

SUSY_HLT_Mu8_TrkIsoVVL_POSTPROCESSING = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_Mu8_TrkIsoVVL_v"),
    verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution     = cms.vstring(""),
    efficiency     = cms.vstring()
)

SUSY_HLT_Mu8 = cms.EDAnalyzer("SUSY_HLT_MuonFakes",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_Mu8_v'),
  TriggerFilter = cms.InputTag('hltL3fL1sMu5L1f0L2f5L3Filtered8', '', 'HLT'), #the last filter in the path                                        
)

SUSY_HLT_Mu8_POSTPROCESSING = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_Mu8_v"),
    verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution     = cms.vstring(""),
    efficiency     = cms.vstring()
)

SUSY_HLT_Mu17_TrkIsoVVL = cms.EDAnalyzer("SUSY_HLT_MuonFakes",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_Mu17_TrkIsoVVL_v'),
  TriggerFilter = cms.InputTag('hltL3fL1sMu1lqL1f0L2f10L3Filtered17TkIsoFiltered0p4', '', 'HLT'), #the last filter in the path                                 
)

SUSY_HLT_Mu17_TrkIsoVVL_POSTPROCESSING = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_Mu17_TrkIsoVVL_v"),
    verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution     = cms.vstring(""),
    efficiency     = cms.vstring()
)

SUSY_HLT_Mu17 = cms.EDAnalyzer("SUSY_HLT_MuonFakes",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_Mu17_v'),
  TriggerFilter = cms.InputTag('hltL3fL1sMu10lqL1f0L2f10L3Filtered17', '', 'HLT'), #the last filter in the path                                 
)

SUSY_HLT_Mu17_POSTPROCESSING = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_Mu17_v"),
    verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution     = cms.vstring(""),
    efficiency     = cms.vstring()
)

SUSY_HLT_TkMu17 = cms.EDAnalyzer("SUSY_HLT_MuonFakes",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_TkMu17_v'),
  TriggerFilter = cms.InputTag('hltL3fL1sMu10lqTkFiltered17Q', '', 'HLT'), #the last filter in the path                                 
)

SUSY_HLT_TkMu17_POSTPROCESSING = DQMEDHarvester("DQMGenericClient",
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

SUSY_HLT_MuonFakes_POSTPROCESSING = cms.Sequence(SUSY_HLT_Mu8_POSTPROCESSING +
                                                 SUSY_HLT_Mu17_POSTPROCESSING+
                                                 SUSY_HLT_TkMu17_POSTPROCESSING+
                                                 SUSY_HLT_Mu8_TrkIsoVVL_POSTPROCESSING+ 
                                                 SUSY_HLT_Mu17_TrkIsoVVL_POSTPROCESSING)
