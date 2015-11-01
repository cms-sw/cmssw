import FWCore.ParameterSet.Config as cms

SUSY_HLT_RazorHbb_Rsq0p02_MR300_2CSV0p7 = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  METCollection = cms.InputTag("pfMet"),
  jetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_Rsq0p02_MR300_TriPFJet80_60_40_DoublePFBTagCSV0p7_Mbb60_200_v'),
  TriggerFilter = cms.InputTag('hltRsq0p02MR300', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltMR200Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)

SUSY_HLT_RazorHbb_Rsq0p02_MR300_2CSV0p7_0p4 = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  METCollection = cms.InputTag("pfMet"),
  jetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_Rsq0p02_MR300_TriPFJet80_60_40_DoublePFBTagCSV0p7_0p4_Mbb60_200_v'),
  TriggerFilter = cms.InputTag('hltRsq0p02MR300', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltMR200Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)

SUSY_HLT_Razor_Main_RsqMR300 = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  METCollection = cms.InputTag("pfMet"),
  jetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_RsqMR300_Rsq0p09_MR200_v'),
  TriggerFilter = cms.InputTag('hltRsqMR300Rsq0p09MR200', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsqMR240Rsq0p0196MR100Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)

SUSY_HLT_Razor_QuadJet_RsqMR300 = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  METCollection = cms.InputTag("pfMet"),
  jetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_RsqMR300_Rsq0p09_MR200_4jet_v'),
  TriggerFilter = cms.InputTag('hltRsqMR300Rsq0p09MR200', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsqMR240Rsq0p0196MR100Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)

SUSY_HLT_Razor_DM_Rsq0p36 = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  METCollection = cms.InputTag("pfMet"),
  jetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_Rsq0p36_v'),
  TriggerFilter = cms.InputTag('hltRsq0p36', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsq0p16Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)

SUSY_HLT_Razor_Main_RsqMR270 = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  METCollection = cms.InputTag("pfMet"),
  jetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_RsqMR270_Rsq0p09_MR200_v'),
  TriggerFilter = cms.InputTag('hltRsqMR270Rsq0p09MR200', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsqMR220Rsq0p0196MR100Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)

SUSY_HLT_Razor_QuadJet_RsqMR270 = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  METCollection = cms.InputTag("pfMet"),
  jetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_RsqMR270_Rsq0p09_MR200_4jet_v'),
  TriggerFilter = cms.InputTag('hltRsqMR270Rsq0p09MR200', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsqMR220Rsq0p0196MR100Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)

SUSY_HLT_Razor_DM_Rsq0p30 = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  METCollection = cms.InputTag("pfMet"),
  jetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_Rsq0p30_v'),
  TriggerFilter = cms.InputTag('hltRsq0p30', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsq0p16Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)

SUSY_HLT_Razor_Main_RsqMR260 = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  METCollection = cms.InputTag("pfMet"),
  jetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_RsqMR260_Rsq0p09_MR200_v'),
  TriggerFilter = cms.InputTag('hltRsqMR260Rsq0p09MR200', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsqMR240Rsq0p0196MR100Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)

SUSY_HLT_Razor_QuadJet_RsqMR260 = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  METCollection = cms.InputTag("pfMet"),
  jetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_RsqMR260_Rsq0p09_MR200_4jet_v'),
  TriggerFilter = cms.InputTag('hltRsqMR260Rsq0p09MR200', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsqMR240Rsq0p0196MR100Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)

SUSY_HLT_Razor_Main_RsqMR240 = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  METCollection = cms.InputTag("pfMet"),
  jetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_RsqMR240_Rsq0p09_MR200_v'),
  TriggerFilter = cms.InputTag('hltRsqMR240Rsq0p09MR200', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsqMR200Rsq0p0196MR100Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)

SUSY_HLT_Razor_QuadJet_RsqMR240 = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  METCollection = cms.InputTag("pfMet"),
  jetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_RsqMR240_Rsq0p09_MR200_4jet_v'),
  TriggerFilter = cms.InputTag('hltRsqMR240Rsq0p09MR200', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsqMR200Rsq0p0196MR100Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)

SUSY_HLT_Razor_DM_Rsq0p25 = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  METCollection = cms.InputTag("pfMet"),
  jetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_Rsq0p25_v'),
  TriggerFilter = cms.InputTag('hltRsq0p25', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsq0p16Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)

SUSY_HLT_Razor_Main_Calo_RsqMR240 = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  METCollection = cms.InputTag("caloMet"),
  jetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_RsqMR240_Rsq0p09_MR200_Calo_v'),
  TriggerFilter = cms.InputTag('hltRsqMR240Rsq0p09MR200Calo', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsqMR240Rsq0p09MR200Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('caloHemispheres')
)

SUSY_HLT_Razor_QuadJet_Calo_RsqMR240 = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  METCollection = cms.InputTag("caloMet"),
  jetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_RsqMR240_Rsq0p09_MR200_4jet_Calo_v'),
  TriggerFilter = cms.InputTag('hltRsqMR240Rsq0p09MR200Calo', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsqMR240Rsq0p09MR200Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('caloHemispheres')
)

SUSY_HLT_Razor_DM_Calo_Rsq0p25 = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  METCollection = cms.InputTag("caloMet"),
  jetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_Rsq0p25_Calo_v'),
  TriggerFilter = cms.InputTag('hltRsq0p25Calo', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsq0p25Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('caloHemispheres')
)

SUSY_HLT_RazorHbb_Rsq0p02_MR300_2CSV0p7_0p4_FASTSIM = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  METCollection = cms.InputTag("pfMet"),
  jetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_Rsq0p02_MR300_TriPFJet80_60_40_DoublePFBTagCSV0p7_0p4_Mbb60_200_v'),
  TriggerFilter = cms.InputTag('hltRsq0p02MR300', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltMR200Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)

SUSY_HLT_RazorHbb_Rsq0p02_MR300_2CSV0p7_FASTSIM = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  METCollection = cms.InputTag("pfMet"),
  jetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_Rsq0p02_MR300_TriPFJet80_60_40_DoublePFBTagCSV0p7_Mbb60_200_v'),
  TriggerFilter = cms.InputTag('hltRsq0p02MR300', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltMR200Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)

SUSY_HLT_Razor_Main_RsqMR300_FASTSIM = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  METCollection = cms.InputTag("pfMet"),
  jetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_RsqMR300_Rsq0p09_MR200_v'),
  TriggerFilter = cms.InputTag('hltRsqMR300Rsq0p09MR200', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsqMR240Rsq0p0196MR100Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)

SUSY_HLT_Razor_QuadJet_RsqMR300_FASTSIM = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  METCollection = cms.InputTag("pfMet"),
  jetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_RsqMR300_Rsq0p09_MR200_4jet_v'),
  TriggerFilter = cms.InputTag('hltRsqMR300Rsq0p09MR200', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsqMR240Rsq0p0196MR100Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)

SUSY_HLT_Razor_DM_Rsq0p36_FASTSIM = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  METCollection = cms.InputTag("pfMet"),
  jetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_Rsq0p36_v'),
  TriggerFilter = cms.InputTag('hltRsq0p36', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsq0p16Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)

SUSY_HLT_Razor_Main_RsqMR270_FASTSIM = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  METCollection = cms.InputTag("pfMet"),
  jetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_RsqMR270_Rsq0p09_MR200_v'),
  TriggerFilter = cms.InputTag('hltRsqMR270Rsq0p09MR200', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsqMR220Rsq0p0196MR100Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)

SUSY_HLT_Razor_QuadJet_RsqMR270_FASTSIM = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  METCollection = cms.InputTag("pfMet"),
  jetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_RsqMR270_Rsq0p09_MR200_4jet_v'),
  TriggerFilter = cms.InputTag('hltRsqMR270Rsq0p09MR200', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsqMR220Rsq0p0196MR100Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)

SUSY_HLT_Razor_DM_Rsq0p30_FASTSIM = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  METCollection = cms.InputTag("pfMet"),
  jetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_Rsq0p30_v'),
  TriggerFilter = cms.InputTag('hltRsq0p30', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsq0p16Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)

SUSY_HLT_Razor_Main_RsqMR260_FASTSIM = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  METCollection = cms.InputTag("pfMet"),
  jetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_RsqMR260_Rsq0p09_MR200_v'),
  TriggerFilter = cms.InputTag('hltRsqMR260Rsq0p09MR200', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsqMR240Rsq0p0196MR100Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)

SUSY_HLT_Razor_QuadJet_RsqMR260_FASTSIM = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  METCollection = cms.InputTag("pfMet"),
  jetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_RsqMR260_Rsq0p09_MR200_4jet_v'),
  TriggerFilter = cms.InputTag('hltRsqMR260Rsq0p09MR200', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsqMR240Rsq0p0196MR100Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)

SUSY_HLT_Razor_Main_RsqMR240_FASTSIM = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  METCollection = cms.InputTag("pfMet"),
  jetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_RsqMR240_Rsq0p09_MR200_v'),
  TriggerFilter = cms.InputTag('hltRsqMR240Rsq0p09MR200', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsqMR200Rsq0p0196MR100Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)

SUSY_HLT_Razor_QuadJet_RsqMR240_FASTSIM = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  METCollection = cms.InputTag("pfMet"),
  jetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_RsqMR240_Rsq0p09_MR200_4jet_v'),
  TriggerFilter = cms.InputTag('hltRsqMR240Rsq0p09MR200', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsqMR200Rsq0p0196MR100Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)

SUSY_HLT_Razor_DM_Rsq0p25_FASTSIM = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  METCollection = cms.InputTag("pfMet"),
  jetCollection = cms.InputTag("ak4PFJetsCHS"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_Rsq0p25_v'),
  TriggerFilter = cms.InputTag('hltRsq0p25', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsq0p16Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('hemispheres')
)


SUSY_HLT_Razor_Main_Calo_RsqMR240_FASTSIM = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  METCollection = cms.InputTag("caloMet"),
  jetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_RsqMR240_Rsq0p09_MR200_Calo_v'),
  TriggerFilter = cms.InputTag('hltRsqMR240Rsq0p09MR200Calo', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsqMR240Rsq0p09MR200Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('caloHemispheres')
)

SUSY_HLT_Razor_QuadJet_Calo_RsqMR240_FASTSIM = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  METCollection = cms.InputTag("caloMet"),
  jetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_RsqMR240_Rsq0p09_MR200_4jet_Calo_v'),
  TriggerFilter = cms.InputTag('hltRsqMR240Rsq0p09MR200Calo', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsqMR240Rsq0p09MR200Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('caloHemispheres')
)

SUSY_HLT_Razor_DM_Calo_Rsq0p25_FASTSIM = cms.EDAnalyzer("SUSY_HLT_Razor",
  trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
  METCollection = cms.InputTag("caloMet"),
  jetCollection = cms.InputTag("ak4CaloJets"),
  TriggerResults = cms.InputTag('TriggerResults','','HLT'),
  TriggerPath = cms.string('HLT_Rsq0p25_Calo_v'),
  TriggerFilter = cms.InputTag('hltRsq0p25Calo', '', 'HLT'), #the last filter in the path
  CaloFilter = cms.InputTag('hltRsq0p25Calo', '', 'HLT'), #filter implementing cuts on calo MR and Rsq
  hemispheres = cms.InputTag('caloHemispheres')
)

SUSY_HLT_Razor_PostVal_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_RsqMR300_Rsq0p09_MR200", "HLT/SUSYBSM/HLT_RsqMR300_Rsq0p09_MR200_4jet", "HLT/SUSYBSM/HLT_Rsq0p36", "HLT/SUSYBSM/HLT_RsqMR270_Rsq0p09_MR200", "HLT/SUSYBSM/HLT_RsqMR270_Rsq0p09_MR200_4jet", "HLT/SUSYBSM/HLT_Rsq0p30", "HLT/SUSYBSM/HLT_RsqMR260_Rsq0p09_MR200", "HLT/SUSYBSM/HLT_RsqMR260_Rsq0p09_MR200_4jet", "HLT/SUSYBSM/HLT_RsqMR240_Rsq0p09_MR200", "HLT/SUSYBSM/HLT_RsqMR240_Rsq0p09_MR200_4jet", "HLT/SUSYBSM/HLT_Rsq0p25", "HLT/SUSYBSM/HLT_RsqMR240_Rsq0p09_MR200_Calo", "HLT/SUSYBSM/HLT_RsqMR240_Rsq0p09_MR200_4jet_Calo", "HLT/SUSYBSM/HLT_Rsq0p25_Calo", "HLT/SUSYBSM/HLT_Rsq0p02_MR300_TriPFJet80_60_40_DoublePFBTagCSV0p7_0p4_Mbb60_200", "HLT/SUSYBSM/HLT_Rsq0p02_MR300_TriPFJet80_60_40_DoublePFBTagCSV0p7_Mbb60_200"),
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
    subDirs        = cms.untracked.vstring("HLT/SUSYBSM/HLT_RsqMR300_Rsq0p09_MR200", "HLT/SUSYBSM/HLT_RsqMR300_Rsq0p09_MR200_4jet", "HLT/SUSYBSM/HLT_Rsq0p36", "HLT/SUSYBSM/HLT_RsqMR270_Rsq0p09_MR200", "HLT/SUSYBSM/HLT_RsqMR270_Rsq0p09_MR200_4jet", "HLT/SUSYBSM/HLT_Rsq0p30", "HLT/SUSYBSM/HLT_RsqMR260_Rsq0p09_MR200", "HLT/SUSYBSM/HLT_RsqMR260_Rsq0p09_MR200_4jet", "HLT/SUSYBSM/HLT_RsqMR240_Rsq0p09_MR200", "HLT/SUSYBSM/HLT_RsqMR240_Rsq0p09_MR200_4jet", "HLT/SUSYBSM/HLT_Rsq0p25", "HLT/SUSYBSM/HLT_RsqMR240_Rsq0p09_MR200_Calo", "HLT/SUSYBSM/HLT_RsqMR240_Rsq0p09_MR200_4jet_Calo", "HLT/SUSYBSM/HLT_Rsq0p25_Calo", "HLT/SUSYBSM/HLT_Rsq0p02_MR300_TriPFJet80_60_40_DoublePFBTagCSV0p7_0p4_Mbb60_200", "HLT/SUSYBSM/HLT_Rsq0p02_MR300_TriPFJet80_60_40_DoublePFBTagCSV0p7_Mbb60_200"),
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


