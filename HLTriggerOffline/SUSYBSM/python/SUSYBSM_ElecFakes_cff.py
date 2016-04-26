import FWCore.ParameterSet.Config as cms

SUSY_HLT_Ele8_IdL_Iso_Jet30 = cms.EDAnalyzer("SUSY_HLT_ElecFakes",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_Ele8_CaloIdL_TrackIdL_IsoVL_PFJet30_v'),
  TriggerFilter = cms.InputTag('hltEle8CaloIdLTrackIdLIsoVLTrackIsoFilter', '', 'HLT'), #the last filter in the path
  TriggerJetFilter = cms.InputTag('hltEle8PFJet30EleCleaned', '', 'HLT'), #the last filter in the path                                      
)
SUSY_HLT_Ele8_IdL_Iso_Jet30_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
    subDirs   = cms.untracked.vstring("HLT/SUSYBSM/HLT_Ele8_CaloIdL_TrackIdL_IsoVL_PFJet30_v"),
    verbose    = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution = cms.vstring(""),
    efficiency = cms.vstring()
)

SUSY_HLT_Ele12_IdL_Iso_Jet30 = cms.EDAnalyzer("SUSY_HLT_ElecFakes",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_Ele12_CaloIdL_TrackIdL_IsoVL_PFJet30_v'),
  TriggerFilter = cms.InputTag('hltEle12CaloIdLTrackIdLIsoVLTrackIsoFilter', '', 'HLT'), #the last filter in the path
  TriggerJetFilter = cms.InputTag('hltEle12PFJet30EleCleaned', '', 'HLT'), #the last filter in the path                                      
)
SUSY_HLT_Ele12_IdL_Iso_Jet30_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
    subDirs   = cms.untracked.vstring("HLT/SUSYBSM/HLT_Ele12_CaloIdL_TrackIdL_IsoVL_PFJet30_v"),
    verbose    = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution = cms.vstring(""),
    efficiency = cms.vstring()
)

SUSY_HLT_Ele17_IdL_Iso_Jet30 = cms.EDAnalyzer("SUSY_HLT_ElecFakes",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_Ele17_CaloIdL_TrackIdL_IsoVL_PFJet30_v'),
  TriggerFilter = cms.InputTag('hltEle17CaloIdLTrackIdLIsoVLTrackIsoFilter', '', 'HLT'), #the last filter in the path
  TriggerJetFilter = cms.InputTag('hltEle17PFJet30EleCleaned', '', 'HLT'), #the last filter in the path
)

SUSY_HLT_Ele17_IdL_Iso_Jet30_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
    subDirs   = cms.untracked.vstring("HLT/SUSYBSM/HLT_Ele17_CaloIdL_TrackIdL_IsoVL_PFJet30_v"),
    verbose    = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution = cms.vstring(""),
    efficiency = cms.vstring()
)

SUSY_HLT_Ele23_IdL_Iso_Jet30 = cms.EDAnalyzer("SUSY_HLT_ElecFakes",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_Ele23_CaloIdL_TrackIdL_IsoVL_PFJet30_v'),
  TriggerFilter = cms.InputTag('hltEle23CaloIdLTrackIdLIsoVLTrackIsoFilter', '', 'HLT'), #the last filter in the path
  TriggerJetFilter = cms.InputTag('hltEle23PFJet30EleCleaned', '', 'HLT'), #the last filter in the path
)

SUSY_HLT_Ele23_IdL_Iso_Jet30_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
    subDirs   = cms.untracked.vstring("HLT/SUSYBSM/HLT_Ele23_CaloIdL_TrackIdL_IsoVL_PFJet30_v"),
    verbose    = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution = cms.vstring(""),
    efficiency = cms.vstring()
)


SUSY_HLT_Ele8_Jet30 = cms.EDAnalyzer("SUSY_HLT_ElecFakes",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_Ele8_CaloIdM_TrackIdM_PFJet30_v'),
  TriggerFilter = cms.InputTag('hltEle8CaloIdMTrkIdMDPhiFilter', '', 'HLT'), #the last filter in the path
  TriggerJetFilter = cms.InputTag('hltEle8NoIsoPFJet30EleCleaned', '', 'HLT'), #the last filter in the path                                      
)
SUSY_HLT_Ele8_Jet30_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
    subDirs   = cms.untracked.vstring("HLT/SUSYBSM/HLT_Ele8_CaloIdM_TrackIdM_PFJet30_v"),
    verbose    = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution = cms.vstring(""),
    efficiency = cms.vstring()
)

SUSY_HLT_Ele12_Jet30 = cms.EDAnalyzer("SUSY_HLT_ElecFakes",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_Ele12_CaloIdM_TrackIdM_PFJet30_v'),
  TriggerFilter = cms.InputTag('hltEle12CaloIdMTrkIdMDPhiFilter', '', 'HLT'), #the last filter in the path
  TriggerJetFilter = cms.InputTag('hltEle12NoIsoPFJet30EleCleaned', '', 'HLT'), #the last filter in the path                                      
)
SUSY_HLT_Ele12_Jet30_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
    subDirs   = cms.untracked.vstring("HLT/SUSYBSM/HLT_Ele12_CaloIdM_TrackIdM_PFJet30_v"),
    verbose    = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution = cms.vstring(""),
    efficiency = cms.vstring()
)

SUSY_HLT_Ele17_Jet30 = cms.EDAnalyzer("SUSY_HLT_ElecFakes",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_Ele17_CaloIdM_TrackIdM_PFJet30_v'),
  TriggerFilter = cms.InputTag('hltEle17CaloIdMTrkIdMDPhiFilter', '', 'HLT'), #the last filter in the path
  TriggerJetFilter = cms.InputTag('hltEle17NoIsoPFJet30EleCleaned', '', 'HLT'), #the last filter in the path                                      
)
SUSY_HLT_Ele17_Jet30_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
    subDirs   = cms.untracked.vstring("HLT/SUSYBSM/HLT_Ele17_CaloIdM_TrackIdM_PFJet30_v"),
    verbose    = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution = cms.vstring(""),
    efficiency = cms.vstring()
)

SUSY_HLT_Ele23_Jet30 = cms.EDAnalyzer("SUSY_HLT_ElecFakes",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  HLTProcess = cms.string('HLT'),
  TriggerPath = cms.string('HLT_Ele23_CaloIdM_TrackIdM_PFJet30_v'),
  TriggerFilter = cms.InputTag('hltEle23CaloIdMTrkIdMDPhiFilter', '', 'HLT'), #the last filter in the path
  TriggerJetFilter = cms.InputTag('hltEle23NoIsoPFJet30EleCleaned', '', 'HLT'), #the last filter in the path                                      
)
SUSY_HLT_Ele23_Jet30_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
    subDirs   = cms.untracked.vstring("HLT/SUSYBSM/HLT_Ele23_CaloIdM_TrackIdM_PFJet30_v"),
    verbose    = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution = cms.vstring(""),
    efficiency = cms.vstring()
)

SUSY_HLT_ElecFakes = cms.Sequence(SUSY_HLT_Ele8_IdL_Iso_Jet30+
                                  SUSY_HLT_Ele12_IdL_Iso_Jet30+
                                  SUSY_HLT_Ele17_IdL_Iso_Jet30+
                                  SUSY_HLT_Ele23_IdL_Iso_Jet30+
                                  SUSY_HLT_Ele8_Jet30+
                                  SUSY_HLT_Ele12_Jet30+
                                  SUSY_HLT_Ele17_Jet30+
                                  SUSY_HLT_Ele23_Jet30)

SUSY_HLT_ElecFakes_POSTPROCESSING = cms.Sequence(SUSY_HLT_Ele8_IdL_Iso_Jet30_POSTPROCESSING+
                                                 SUSY_HLT_Ele12_IdL_Iso_Jet30_POSTPROCESSING+
                                                 SUSY_HLT_Ele17_IdL_Iso_Jet30_POSTPROCESSING+
                                                 SUSY_HLT_Ele23_IdL_Iso_Jet30_POSTPROCESSING+
                                                 SUSY_HLT_Ele8_Jet30_POSTPROCESSING+
                                                 SUSY_HLT_Ele12_Jet30_POSTPROCESSING+
                                                 SUSY_HLT_Ele17_Jet30_POSTPROCESSING+
                                                 SUSY_HLT_Ele23_Jet30_POSTPROCESSING)
