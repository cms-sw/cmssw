import FWCore.ParameterSet.Config as cms

jetMETHLTOfflineSource = cms.EDFilter("JetMETHLTOfflineSource",
 
                                 #triggerTriggerEvent_hltTriggerSummaryAOD_HLT
                                 triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
                                 #
                                 # edmTriggerResults_TriggerResults_RECO
                                 triggerResultsLabel = cms.InputTag("TriggerResults::HLT"),
                                 processname = cms.string("HLT"),

                                 L1Taus  = cms.InputTag("hltL1extraParticles","Tau"),
                                 L1CJets = cms.InputTag("hltL1extraParticles","Central"),
                                 L1FJets = cms.InputTag("hltL1extraParticles","Forward"),

                                 CaloMETCollectionLabel = cms.InputTag("met"),
                                 CaloJetCollectionLabel = cms.InputTag("iterativeCone5CaloJets"),
                                 #CaloJetCollectionLabel = cms.InputTag("L2L3CorJetIC5Calo"),

                                 #-----
                                 #
                                 # Numerator path name : Numerator path name : HLT or L1s : Trigger threshold
                                 #     
                                 #HLTPathsEffSingleJet = cms.vstring("HLT_L1Jet15:HLT_MinBiasPixel",
                                 #                            "HLT_Jet30:HLT_L1Jet15",
                                 #                            "HLT_Jet50:HLT_Jet30",
                                 #                            "HLT_Jet80:HLT_Jet50",
                                 #                            "HLT_Jet110:HLT_Jet80",
                                 #                            "HLT_Jet180:HLT_Jet110"),
                                 HLTPathsEffSingleJet = cms.vstring(
                                                              "Jet15U:L1Jet6U:HLT:15",
                                                              "Jet30U:Jet15U:HLT:30",
                                                              "Jet50U:Jet30U:HLT:50",
                                                              "Jet30U:L1Jet6U:L1s:15",
                                                              "Jet50U:Jet15U:L1s:30",
                                                              #
                                                              "Jet30:L1Jet15:HLT:30",
                                                              "Jet50:Jet30:HLT:50",
                                                              "Jet80:Jet50:HLT:80",
                                                              "Jet110:Jet80:HLT:110",
                                                              "Jet180:Jet110:HLT:180",
                                                              "Jet50:L1Jet15:L1s:30",
                                                              "Jet80:Jet30:L1s:50",
                                                              "Jet110:Jet50:L1s:70",
                                                              "Jet180:Jet50:L1s:70"),

                                 #HLTPathsEffDiJetAve  = cms.vstring("HLT_L1Jet15:HLT_MinBiasPixel",
                                 #                            "HLT_DiJetAve15U_1E31:HLT_L1Jet15",
                                 #                            "HLT_DiJetAve30U_1E31:HLT_DiJetAve15U_1E31",
                                 #                            "HLT_DiJetAve50U:HLT_DiJetAve30U_1E31",
                                 #                            "HLT_DiJetAve70U:HLT_DiJetAve50U",
                                 #                            "HLT_DiJetAve130U:HLT_DiJetAve70U"),
                                 HLTPathsEffDiJetAve  = cms.vstring("DiJetAve15U_8E29:L1Jet6U:HLT:15",
                                                                    "DiJetAve30U_8E29:DiJetAve15U_8E29:HLT:30",
                                                                    #
                                                                    "DiJetAve15U_1E31:L1Jet6U:HLT:15",
                                                                    "DiJetAve30U_1E31:DiJetAve15U_1E31:HLT:30",
                                                                    "DiJetAve50U:DiJetAve30U_1E31:HLT:50",
                                                                    "DiJetAve70U:DiJetAve50U:HLT:70",
                                                                    "DiJetAve130U:DiJetAve70U:HLT:130"),

                                 HLTPathsEffMET       = cms.vstring(
                                                             "MET35:L1MET20",
                                                             "MET100:MET35",
                                                             "MET25:L1MET20",
                                                             "MET50:MET25"
                                                             "MET100:MET50"),

                                 HLTPathsEffMHT       = cms.vstring("HT300_MHT100:HT250"),

                                 #-----

                                 HLTPathsMonSingleJet = cms.vstring("L1Jet6U",
                                                             "Jet30U",
                                                             "Jet50U",
                                                             "L1Jet15",
                                                             "Jet30",
                                                             "Jet50",
                                                             "Jet80",
                                                             "Jet110",
                                                             "Jet180"),

                                 HLTPathsMonDiJetAve  = cms.vstring(
                                                             "DiJetAve15U_8E29",
                                                             "DiJetAve30U_8E29",
                                                             "DiJetAve15U_1E31",
                                                             "DiJetAve30U_1E31",
                                                             "DiJetAve50U",
                                                             "DiJetAve70U",
                                                             "DiJetAve130U"
                                                             ),

                                 HLTPathsMonMET       = cms.vstring("L1MET20",
                                                             "MET25",
                                                             "MET35",
                                                             "MET50",
                                                             "MET100"),

                                 HLTPathsMonMHT       = cms.vstring("HT250"),

                                 #-----
                                      
                                 DQMDirName=cms.string("HLT/JetMET"),

                                 hltTag = cms.string("HLT")

                                 #-----
                                 
)


