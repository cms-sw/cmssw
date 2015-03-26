import FWCore.ParameterSet.Config as cms


SUSY_HLT_Razor_Main = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  pfMETCollection = cms.InputTag("pfMet"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_RsqMR300_Rsq0p09_MR200_v'),
  TriggerFilter = cms.InputTag('hltRsqMR300Rsq0p09MR200', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsqMR240Rsq0p0196MR100Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)

SUSY_HLT_Razor_QuadJet = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  pfMETCollection = cms.InputTag("pfMet"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_RsqMR300_Rsq0p09_MR200_4jet_v'),
  TriggerFilter = cms.InputTag('hltRsqMR300Rsq0p09MR200', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsqMR240Rsq0p0196MR100Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)

SUSY_HLT_Razor_DM = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  pfMETCollection = cms.InputTag("pfMet"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_Rsq0p36_v'),
  TriggerFilter = cms.InputTag('hltRsq0p36', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsq0p16Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)

SUSY_HLT_Razor_Main_7e33 = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  pfMETCollection = cms.InputTag("pfMet"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_RsqMR260_Rsq0p09_MR200_v'),
  TriggerFilter = cms.InputTag('hltRsqMR260Rsq0p09MR200', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsqMR240Rsq0p0196MR100Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)

SUSY_HLT_Razor_QuadJet_7e33 = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  pfMETCollection = cms.InputTag("pfMet"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_RsqMR260_Rsq0p09_MR200_4jet_v'),
  TriggerFilter = cms.InputTag('hltRsqMR260Rsq0p09MR200', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsqMR240Rsq0p0196MR100Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)



SUSY_HLT_Razor_Main_FASTSIM = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  pfMETCollection = cms.InputTag("pfMet"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_RsqMR300_Rsq0p09_MR200_v'),
  TriggerFilter = cms.InputTag('hltRsqMR300Rsq0p09MR200', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsqMR240Rsq0p0196MR100Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)

SUSY_HLT_Razor_QuadJet_FASTSIM = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  pfMETCollection = cms.InputTag("pfMet"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_RsqMR300_Rsq0p09_MR200_4jet_v'),
  TriggerFilter = cms.InputTag('hltRsqMR300Rsq0p09MR200', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsqMR240Rsq0p0196MR100Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)

SUSY_HLT_Razor_DM_FASTSIM = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  pfMETCollection = cms.InputTag("pfMet"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_Rsq0p36_v'),
  TriggerFilter = cms.InputTag('hltRsq0p36', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsq0p16Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)

SUSY_HLT_Razor_Main_7e33_FASTSIM = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  pfMETCollection = cms.InputTag("pfMet"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_RsqMR260_Rsq0p09_MR200_v'),
  TriggerFilter = cms.InputTag('hltRsqMR260Rsq0p09MR200', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsqMR240Rsq0p0196MR100Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)

SUSY_HLT_Razor_QuadJet_7e33_FASTSIM = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  pfMETCollection = cms.InputTag("pfMet"),
  pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_RsqMR260_Rsq0p09_MR200_4jet_v'),
  TriggerFilter = cms.InputTag('hltRsqMR260Rsq0p09MR200', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsqMR240Rsq0p0196MR100Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)



SUSY_HLT_Razor_PostVal_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_RsqMR300_Rsq0p09_MR200", "HLT/SUSYBSM/HLT_RsqMR300_Rsq0p09_MR200_4jet", "HLT/SUSYBSM/HLT_Rsq0p36", "HLT/SUSYBSM/HLT_RsqMR260_Rsq0p09_MR200", "HLT/SUSYBSM/HLT_RsqMR260_Rsq0p09_MR200_4jet"),
    verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution     = cms.vstring(""),
    efficiency     = cms.vstring(
       "mr_turnOn_eff 'M_{R} efficiency (R^{2} > 0.15); M_{R} (GeV); #epsilon' mr mr_denom",
       "mr_tight_turnOn_eff 'M_{R} efficiency (R^{2} > 0.25); M_{R} (GeV); #epsilon' mr_tight mr_tight_denom",
       "rsq_turnOn_eff 'R^{2} efficiency (M_{R} > 300); R^{2}; #epsilon' rsq rsq_denom",
       "rsq_tight_turnOn_eff 'R^{2} efficiency (M_{R} > 400); R^{2}; #epsilon' rsq_tight rsq_tight_denom",
       "rsq_loose_turnOn_eff 'R^{2} efficiency (M_{R} > 0); R^{2}; #epsilon' rsq_loose rsq_loose_denom",
       "mrRsq_turnOn_eff '2D efficiency; M_{R} (GeV); R^{2}; #epsilon' mrRsq mrRsq_denom"
    )
)


SUSY_HLT_Razor_PostVal_FASTSIM_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_RsqMR300_Rsq0p09_MR200", "HLT/SUSYBSM/HLT_RsqMR300_Rsq0p09_MR200_4jet", "HLT/SUSYBSM/HLT_Rsq0p36", "HLT/SUSYBSM/HLT_RsqMR260_Rsq0p09_MR200", "HLT/SUSYBSM/HLT_RsqMR260_Rsq0p09_MR200_4jet"),
    verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
    resolution     = cms.vstring(""),
    efficiency     = cms.vstring(
       "mr_turnOn_eff 'M_{R} efficiency (R^{2} > 0.15); M_{R} (GeV); #epsilon' mr mr_denom",
       "mr_tight_turnOn_eff 'M_{R} efficiency (R^{2} > 0.25); M_{R} (GeV); #epsilon' mr_tight mr_tight_denom",
       "rsq_turnOn_eff 'R^{2} efficiency (M_{R} > 300); R^{2}; #epsilon' rsq rsq_denom",
       "rsq_tight_turnOn_eff 'R^{2} efficiency (M_{R} > 400); R^{2}; #epsilon' rsq_tight rsq_tight_denom",
       "rsq_loose_turnOn_eff 'R^{2} efficiency (M_{R} > 0); R^{2}; #epsilon' rsq_loose rsq_loose_denom",
       "mrRsq_turnOn_eff '2D efficiency; M_{R} (GeV); R^{2}; #epsilon' mrRsq mrRsq_denom"
    )
)




