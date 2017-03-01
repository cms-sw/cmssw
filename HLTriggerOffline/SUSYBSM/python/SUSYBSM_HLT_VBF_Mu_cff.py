import FWCore.ParameterSet.Config as cms

SUSY_HLT_Mu10_VBF = cms.EDAnalyzer("SUSY_HLT_VBF_Mu10",
                                 trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'), #to use with test sample
                                 #trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
                                 MuonCollection = cms.InputTag("muons"),
                                 pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
                                 caloJetCollection = cms.InputTag("ak4CaloJets"),
                                 pfMETCollection = cms.InputTag("pfMet"),
                                 caloMETCollection = cms.InputTag("caloMet"), 
                                 TriggerResults = cms.InputTag('TriggerResults','','HLT'), #to use with test sample
                                 HLTProcess = cms.string('HLT'),
                                 TriggerPath = cms.string('HLT_Mu10_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT350_PFMETNoMu60_v'),
                                 TriggerFilterMuon  = cms.InputTag('hltMuon10RelTrkIsoVVLFiltered0p4','','HLT'),
                                 TriggerFilterMJJ  = cms.InputTag('hltDiPFJet40MJJ750DEta3p5','','HLT'),
                                 TriggerFilterHT = cms.InputTag('hltPFHT350Jet30','','HLT'),
                                 TriggerFilterMET  = cms.InputTag('hltPFMETNoMu60','','HLT'),
                                 TriggerFilterCaloMET  = cms.InputTag('hltMETClean10','','HLT'),
 # hltMETCleanUsingJetID20','','HLT'),
                                 PtThrJet = cms.untracked.double(30.0),
                                 EtaThrJet = cms.untracked.double(3.0),
                                 PtThrJetTrig  = cms.untracked.double(30.0),
                                 EtaThrJetTrig  = cms.untracked.double(5.0),
                                 DeltaEtaVBFJets  = cms.untracked.double(3.5),
                                 PFMetCutOnline  = cms.untracked.double(60.0),
                                 MuonCutOnline  = cms.untracked.double(10.0),
                                 HTCutOnline = cms.untracked.double(350.0),
                                 MJJCutOnline = cms.untracked.double(750.0)
                                 
                                 )

SUSY_HLT_Mu10_VBF_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
                                                subDirs        = cms.untracked.vstring("HLT/SUSYBSM/SUSY_HLT_VBF_Mu10_v"),
                                                verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
                                                resolution     = cms.vstring(""),
                                                efficiency     = cms.vstring(
        "MuonTurnOn_eff 'Turn-on vs Mu pT; pT (GeV); #epsilon' h_num_muonpt h_den_muonpt",
        "MJJTurnOn_eff 'Turn-on vs Mjj; Mjj (GeV); #epsilon' h_num_mjj h_den_mjj",
        "pfHTTurnOn_eff 'Turn-on vs pf HT; pf HT (GeV); #epsilon' h_num_ht h_den_ht",
        "pfMETTurnOn_eff 'Turn-on vs pf MET; MET (GeV) ; #epsilon' h_num_met h_den_met", 
        )
                                                )

SUSY_HLT_Mu8_VBF = cms.EDAnalyzer("SUSY_HLT_VBF_Mu8",
                                 trigSummary = cms.InputTag("hltTriggerSummaryAOD",'', 'HLT'), #to use with test sample
                                 #trigSummary = cms.InputTag("hltTriggerSummaryAOD"),
                                 MuonCollection = cms.InputTag("muons"),
                                 pfJetCollection = cms.InputTag("ak4PFJetsCHS"),
                                 caloJetCollection = cms.InputTag("ak4CaloJets"),
                                 pfMETCollection = cms.InputTag("pfMet"),
                                 caloMETCollection = cms.InputTag("caloMet"), 
                                 TriggerResults = cms.InputTag('TriggerResults','','HLT'), #to use with test sample
                                 HLTProcess = cms.string('HLT'),
                                 TriggerPath = cms.string('HLT_Mu8_TrkIsoVVL_DiPFJet40_DEta3p5_MJJ750_HTT300_PFMETNoMu60_v'),
                                 TriggerFilterMuon  = cms.InputTag('hltMuon8RelTrkIsoVVLFiltered0p4','','HLT'),
                                 TriggerFilterMJJ  = cms.InputTag('hltDiPFJet40MJJ750DEta3p5','','HLT'),
                                 TriggerFilterHT = cms.InputTag('hltPFHT300Jet30','','HLT'),
                                 TriggerFilterMET  = cms.InputTag('hltPFMETNoMu60','','HLT'),
                                 TriggerFilterCaloMET  = cms.InputTag('hltMETClean10','','HLT'),
 # hltMETCleanUsingJetID20','','HLT'),
                                 PtThrJet = cms.untracked.double(30.0),
                                 EtaThrJet = cms.untracked.double(3.0),
                                 PtThrJetTrig  = cms.untracked.double(30.0),
                                 EtaThrJetTrig  = cms.untracked.double(5.0),
                                 DeltaEtaVBFJets  = cms.untracked.double(3.5),
                                 PFMetCutOnline  = cms.untracked.double(60.0),
                                 MuonCutOnline  = cms.untracked.double(8.0),
                                 HTCutOnline = cms.untracked.double(300.0),
                                 MJJCutOnline = cms.untracked.double(750.0)
                                 
                                 )

SUSY_HLT_Mu8_VBF_POSTPROCESSING = cms.EDAnalyzer("DQMGenericClient",
                                                subDirs        = cms.untracked.vstring("HLT/SUSYBSM/SUSY_HLT_VBF_Mu8_v"),
                                                verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
                                                resolution     = cms.vstring(""),
                                                efficiency     = cms.vstring(
        "MuonTurnOn_eff 'Turn-on vs Mu pT; pT (GeV); #epsilon' h_num_muonpt h_den_muonpt",
        "MJJTurnOn_eff 'Turn-on vs Mjj; Mjj (GeV); #epsilon' h_num_mjj h_den_mjj",
        "pfHTTurnOn_eff 'Turn-on vs pf HT; pf HT (GeV); #epsilon' h_num_ht h_den_ht",
        "pfMETTurnOn_eff 'Turn-on vs pf MET; MET (GeV) ; #epsilon' h_num_met h_den_met", 
        )
                                                )

SUSY_HLT_Mu_VBF = cms.Sequence( SUSY_HLT_Mu10_VBF +
                                SUSY_HLT_Mu8_VBF
                                )

SUSY_HLT_Mu_VBF_POSTPROCESSING = cms.Sequence( SUSY_HLT_Mu10_VBF_POSTPROCESSING +
                                               SUSY_HLT_Mu8_VBF_POSTPROCESSING
                                               )
