import FWCore.ParameterSet.Config as cms

jetMETHLTOfflineSource = cms.EDFilter("JetMETHLTOfflineSource",

                                 triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
                                 triggerResultsLabel = cms.InputTag("TriggerResults::HLT"),
                                 processname = cms.string("HLT"),

                                 CaloMETCollectionLabel = cms.InputTag("met"),
                                 CaloJetCollectionLabel = cms.InputTag("iterativeCone5CaloJets"),
                                 #CaloJetCollectionLabel = cms.InputTag("L2L3CorJetIC5Calo"),

                                 #-----

                                 #HLTPathsEffSingleJet = cms.vstring("HLT_L1Jet15:HLT_MinBiasPixel",
                                 #                            "HLT_Jet30:HLT_L1Jet15",
                                 #                            "HLT_Jet50:HLT_Jet30",
                                 #                            "HLT_Jet80:HLT_Jet50",
                                 #                            "HLT_Jet110:HLT_Jet80",
                                 #                            "HLT_Jet180:HLT_Jet110"),
                                 HLTPathsEffSingleJet = cms.vstring(
                                                             "Jet15U:L1Jet6U",
                                                             "Jet30U:Jet15U",
                                                             "Jet50U:Jet30U"),

                                 #HLTPathsEffDiJetAve  = cms.vstring("HLT_L1Jet15:HLT_MinBiasPixel",
                                 #                            "HLT_DiJetAve15U_1E31:HLT_L1Jet15",
                                 #                            "HLT_DiJetAve30U_1E31:HLT_DiJetAve15U_1E31",
                                 #                            "HLT_DiJetAve50U:HLT_DiJetAve30U_1E31",
                                 #                            "HLT_DiJetAve70U:HLT_DiJetAve50U",
                                 #                            "HLT_DiJetAve130U:HLT_DiJetAve70U"),
                                 HLTPathsEffDiJetAve  = cms.vstring("DiJetAve15U_8E29:L1Jet6U",
                                                             "DiJetAve30U_8E29:DiJetAve15U_8E29"),

                                 HLTPathsEffMET       = cms.vstring(
                                                             "MET35:L1MET20",
                                                             "MET100:MET35"),

                                 HLTPathsEffMHT       = cms.vstring("HT300_MHT100:HT250"),

                                 #-----

                                 HLTPathsMonSingleJet = cms.vstring("L1Jet6U",
                                                             "Jet30U",
                                                             "Jet50U"),

                                 HLTPathsMonDiJetAve  = cms.vstring(
                                                             "DiJetAve15U_8E29",
                                                             "DiJetAve30U_8E29"),

                                 HLTPathsMonMET       = cms.vstring("L1MET20",
                                                             "MET35",
                                                             "MET100"),

                                 HLTPathsMonMHT       = cms.vstring("HT250"),

                                 #-----
                                      
                                 DQMDirName=cms.string("HLT/JetMET"),

                                 hltTag = cms.string("HLT")

                                 #-----
                                 
)


