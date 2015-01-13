import FWCore.ParameterSet.Config as cms

def overrideCentrality(process):
    process.GlobalTag.toGet.extend([

                                    #==================== MC Tables ====================
                                    cms.PSet(record = cms.string("HeavyIonRcd"),
                                             tag = cms.string("CentralityTable_HFhits40_AMPTOrgan_v0_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("HFhitsAMPT_Organ")
                                             ),
                                    cms.PSet(record = cms.string("HeavyIonRcd"),
                                             tag = cms.string("CentralityTable_PixelHits40_AMPTOrgan_v0_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("PixelHitsAMPT_Organ")
                                             ),

                                    cms.PSet(record = cms.string("HeavyIonRcd"),
                                             tag = cms.string("CentralityTable_HFhits40_HydjetBass_vv44x04_mc"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("HFhitsHydjet_Bass")
                                             ),
                                    cms.PSet(record = cms.string("HeavyIonRcd"),
                                             tag = cms.string("CentralityTable_PixelHits40_HydjetBass_vv44x04_mc"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("PixelHitsHydjet_Bass")
                                             ),
                                    cms.PSet(record = cms.string("HeavyIonRcd"),
                                             tag = cms.string("CentralityTable_Tracks40_HydjetBass_vv44x04_mc"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("TracksHydjet_Bass")
                                             ),
                                    cms.PSet(record = cms.string("HeavyIonRcd"),
                                             tag = cms.string("CentralityTable_PixelTracks40_HydjetBass_vv44x04_mc"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("PixelTracksHydjet_Bass")
                                             ),

                                    cms.PSet(record = cms.string("HeavyIonRcd"),
                                             tag = cms.string("CentralityTable_HFtowers40_HydjetBass_vv44x04_mc"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("HFtowersHydjet_Bass")
                                             ),

                                    cms.PSet(record = cms.string("HeavyIonRcd"),
                                             tag = cms.string("CentralityTable_HFhits40_HydjetDrum_vv44x05_mc"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("HFhitsHydjet_Drum")
                                             ),
                                    cms.PSet(record = cms.string("HeavyIonRcd"),
                                             tag = cms.string("CentralityTable_PixelHits40_HydjetDrum_vv44x05_mc"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("PixelHitsHydjet_Drum")
                                             ),
                                    cms.PSet(record = cms.string("HeavyIonRcd"),
                                             tag = cms.string("CentralityTable_Tracks40_HydjetDrum_vv44x05_mc"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("TracksHydjet_Drum")
                                             ),
                                    cms.PSet(record = cms.string("HeavyIonRcd"),
                                             tag = cms.string("CentralityTable_PixelTracks40_HydjetDrum_vv44x05_mc"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("PixelTracksHydjet_Drum")
                                             ),

                                    cms.PSet(record = cms.string("HeavyIonRcd"),
                 tag = cms.string("CentralityTable_HFtowers200_HydjetDrum_v5315x01_mc"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("HFtowersHydjet_Drum")
                                             ),

        cms.PSet(record = cms.string("HeavyIonRcd"),
                 tag = cms.string("CentralityTable_PixelHits40_Glauber2010A_v3_effB_offline"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("PixelHits")
             ),

        cms.PSet(record = cms.string("HeavyIonRcd"),
                 tag = cms.string("CentralityTable_HFhits40_Glauber2010A_v3_effB_offline"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("HFhits")
             ),

        cms.PSet(record = cms.string("HeavyIonRcd"),
                 tag = cms.string("CentralityTable_HFtowers200_Glauber2010A_v5315x01_offline"),
                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("HFtowers")
             ),

                                    #==================== pPb data taking 2013 =====================================

                                    cms.PSet(record = cms.string("HeavyIonRcd"),
                                             tag = cms.string("CentralityTable_HFplus100_PA2012B_v533x01_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("HFtowersPlusTrunc")
                                             ),
                                    cms.PSet(record = cms.string("HeavyIonRcd"),
                                             tag = cms.string("CentralityTable_HFtrunc100_PA2012B_v538x02_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("HFtowersTrunc")
                                             ),

                                    #==================== pPb MC 2013 =====================================

                                    cms.PSet(record = cms.string("HeavyIonRcd"),
                                             tag = cms.string("CentralityTable_Tracks100_HijingPA_v538x02_mc"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("TracksHijing")
                                             ),
                                    cms.PSet(record = cms.string("HeavyIonRcd"),
                                             tag = cms.string("CentralityTable_HFplus100_HijingPA_v538x02_mc"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("HFtowersPlusTruncHijing")
                                             ),
                                    cms.PSet(record = cms.string("HeavyIonRcd"),
                                             tag = cms.string("CentralityTable_HFtrunc100_HijingPA_v538x01_mc"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("HFtowersTruncHijing")
                                             ),
                                    cms.PSet(record = cms.string("HeavyIonRcd"),
                                             tag = cms.string("CentralityTable_HFplus100_EposPA_v538x01_mc"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("HFtowersPlusTruncEpos")
                                             ),
                                    cms.PSet(record = cms.string("HeavyIonRcd"),
                                             tag = cms.string("CentralityTable_HFtrunc100_EposPA_v538x01_mc"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("HFtowersTruncEpos")
                                             ),

                                    ])
    return process

def overrideJEC_pPb5020(process):
    process.GlobalTag.toGet.extend([
## no Pu
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JEC_gammajet_HiWinter13_STARTHI53_LV1_5_3_20_AK3Calo"),
                 connect = cms.untracked.string("sqlite_file:JEC_GAMMAJET_v1_STARTHI53_LV1.db"),
                 label = cms.untracked.string("AK3Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JEC_gammajet_HiWinter13_STARTHI53_LV1_5_3_20_AK4Calo"),
                 connect = cms.untracked.string("sqlite_file:JEC_GAMMAJET_v1_STARTHI53_LV1.db"),
                 label = cms.untracked.string("AK4Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JEC_gammajet_HiWinter13_STARTHI53_LV1_5_3_20_AK5Calo"),
                 connect = cms.untracked.string("sqlite_file:JEC_GAMMAJET_v1_STARTHI53_LV1.db"),
                 label = cms.untracked.string("AK5Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JEC_gammajet_HiWinter13_STARTHI53_LV1_5_3_20_AK3PF"),
                 connect = cms.untracked.string("sqlite_file:JEC_GAMMAJET_v1_STARTHI53_LV1.db"),
                 label = cms.untracked.string("AK3PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JEC_gammajet_HiWinter13_STARTHI53_LV1_5_3_20_AK4PF"),
                 connect = cms.untracked.string("sqlite_file:JEC_GAMMAJET_v1_STARTHI53_LV1.db"),
                 label = cms.untracked.string("AK4PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JEC_gammajet_HiWinter13_STARTHI53_LV1_5_3_20_AK5PF"),
                 connect = cms.untracked.string("sqlite_file:JEC_GAMMAJET_v1_STARTHI53_LV1.db"),
                 label = cms.untracked.string("AK5PF_generalTracks")
             ),
## Pu
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JEC_gammajet_HiWinter13_STARTHI53_LV1_5_3_20_AK3Calo"),
                 connect = cms.untracked.string("sqlite_file:JEC_GAMMAJET_v1_STARTHI53_LV1.db"),
                 label = cms.untracked.string("AKPu3Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JEC_gammajet_HiWinter13_STARTHI53_LV1_5_3_20_AK4Calo"),
                 connect = cms.untracked.string("sqlite_file:JEC_GAMMAJET_v1_STARTHI53_LV1.db"),
                 label = cms.untracked.string("AKPu4Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JEC_gammajet_HiWinter13_STARTHI53_LV1_5_3_20_AK5Calo"),
                 connect = cms.untracked.string("sqlite_file:JEC_GAMMAJET_v1_STARTHI53_LV1.db"),
                 label = cms.untracked.string("AKPu5Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JEC_gammajet_HiWinter13_STARTHI53_LV1_5_3_20_AK3PF"),
                 connect = cms.untracked.string("sqlite_file:JEC_GAMMAJET_v1_STARTHI53_LV1.db"),
                 label = cms.untracked.string("AKPu3PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JEC_gammajet_HiWinter13_STARTHI53_LV1_5_3_20_AK4PF"),
                 connect = cms.untracked.string("sqlite_file:JEC_GAMMAJET_v1_STARTHI53_LV1.db"),
                 label = cms.untracked.string("AKPu4PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JEC_gammajet_HiWinter13_STARTHI53_LV1_5_3_20_AK5PF"),
                 connect = cms.untracked.string("sqlite_file:JEC_GAMMAJET_v1_STARTHI53_LV1.db"),
                 label = cms.untracked.string("AKPu5PF_generalTracks")
             ),
    ])
    return process

#======  Final default common functions including centrality

def overrideGT_pPb5020(process):
    overrideCentrality(process)
    overrideJEC_pPb5020(process)
    return process
