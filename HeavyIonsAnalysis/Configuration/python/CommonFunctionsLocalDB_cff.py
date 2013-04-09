
import FWCore.ParameterSet.Config as cms

# Turn of MC dependence in pat sequence
def removePatMCMatch(process):
    process.prod.remove(process.genPartons)
    process.prod.remove(process.heavyIonCleanedGenJets)
    process.prod.remove(process.hiPartons)
    process.prod.remove(process.patJetGenJetMatch)
    process.prod.remove(process.patJetPartonMatch)
    
    process.patJets.addGenPartonMatch   = False
    process.patJets.embedGenPartonMatch = False
    process.patJets.genPartonMatch      = ''
    process.patJets.addGenJetMatch      = False
    process.patJets.genJetMatch      = ''
    process.patJets.getJetMCFlavour     = False
    process.patJets.JetPartonMapSource  = ''
    return process

# Top Config to turn off all mc dependence
def disableMC(process):
    process.prod.remove(process.heavyIon)
    removePatMCMatch(process)
    return process

def hltFromREDIGI(process):
    process.hltanalysis.HLTProcessName      = "REDIGI"
    process.hltanalysis.l1GtObjectMapRecord = cms.InputTag("hltL1GtObjectMap::REDIGI")
    process.hltanalysis.l1GtReadoutRecord   = cms.InputTag("hltGtDigis::REDIGI")
    process.hltanalysis.hltresults          = cms.InputTag("TriggerResults::REDIGI")
    return process

def overrideBeamSpot(process):
    process.GlobalTag.toGet = cms.VPSet(
                                        cms.PSet(record = cms.string("BeamSpotObjectsRcd"),
                                                 tag = cms.string("Realistic2.76ATeVCollisions_STARTUP_v0_mc"),
                                                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_BEAMSPOT")
                                                 ),
                                        )
    return process


def addRPFlat(process):
    process.GlobalTag.toGet.extend([
                                    cms.PSet(record = cms.string("HeavyIonRPRcd"),
                                             tag = cms.string("RPFlatParams_Test_v0_offline"),
                                             connect = cms.untracked.string("frontier://FrontierPrep/CMS_COND_TEMP"),
                                             ),
                                    ])
    return process


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
                                             tag = cms.string("CentralityTable_HFtowers40_HydjetDrum_vv44x05_mc"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("HFtowersHydjet_Drum")
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

def overrideJEC_PbPb2760(process):
    process.GlobalTag.toGet.extend([
                                    
                                    #==================== JET CORRECTIONS
                                    
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_AK1PFTowers_hiIterativeTracks_HI_2760GeV_v3_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK1PF_hiIterativeTracks")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_AK2PFTowers_hiIterativeTracks_HI_2760GeV_v3_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK2PF_hiIterativeTracks")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_AK3PFTowers_hiIterativeTracks_HI_2760GeV_v3_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK3PF_hiIterativeTracks")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_AK4PFTowers_hiIterativeTracks_HI_2760GeV_v3_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK4PF_hiIterativeTracks")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_AK5PFTowers_hiIterativeTracks_HI_2760GeV_v3_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK5PF_hiIterativeTracks")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_AK6PFTowers_hiIterativeTracks_HI_2760GeV_v3_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK6PF_hiIterativeTracks")
                                             ),
                                    
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_IC5Calo_2760GeV_v0_offline"),
                                             # pp 7TeV version:       JetCorrectorParametersCollection_Fall12_V5_DATA_IC5Calo
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("IC5Calo_2760GeV")
                                             ),
                                    ])
    return process

def overrideJEC_pPb5020(process):
    process.GlobalTag.toGet.extend([
                                    
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_5TeV_538_v03_AK1Calo_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK1Calo_HI")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_5TeV_538_v04_AK2Calo_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK2Calo_HI")
                                             ),
                                    
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_5TeV_538_v05_AK3Calo_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK3Calo_HI")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_5TeV_538_v05_AK4Calo_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK4Calo_HI")
                                             ),
                                    
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_5TeV_538_v05_AK5Calo_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK5Calo_HI")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_5TeV_538_v04_AK6Calo_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK6Calo_HI")
                                             ),
                                    
                                    
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v02_AK2PF_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK1PF_generalTracks")
                                             ),
                                    
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v04_AK2PF_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK2PF_generalTracks")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK3PF_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK3PF_generalTracks")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK4PF_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK4PF_generalTracks")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK5PF_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK5PF_generalTracks")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v04_AK6PF_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK6PF_generalTracks")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_Fall12_V5_DATA_AK7PF"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK7PF_generalTracks")
                                             ),
                                    
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_5TeV_538_v03_AK1Calo_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AKPu1Calo_HI")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_5TeV_538_v04_AK2Calo_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AKPu2Calo_HI")
                                             ),
                                    
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_5TeV_538_v08_AKPu3Calo_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AKPu3Calo_HI")
                                             ),
                                    
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_5TeV_538_v08_AKPu4Calo_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AKPu4Calo_HI")
                                             ),
                                    
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_5TeV_538_v08_AKPu5Calo_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AKPu5Calo_HI")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_5TeV_538_v04_AK6Calo_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AKPu6Calo_HI")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v02_AK2PF_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AKPu1PF_generalTracks")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v04_AK2PF_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AKPu2PF_generalTracks")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v08_AKPu3PF_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AKPu3PF_generalTracks")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v08_AKPu4PF_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AKPu4PF_generalTracks")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v08_AKPu5PF_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AKPu5PF_generalTracks")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v04_AK6PF_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AKPu6PF_generalTracks")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_Fall12_V5_DATA_AK7PF"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AKPu7PF_generalTracks")
                                             ),
                                    ])
    
    return process


def overrideJEC_pp2760(process):
    process.GlobalTag.toGet.extend([
                                    
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_5TeV_538_v03_AK1Calo_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK1Calo_HI")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_5TeV_538_v04_AK2Calo_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK2Calo_HI")
                                             ),
                                    
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             #                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_538_v07_AK3Calo_offline"),
                                             #                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_PythiaZ2_2760GeV_538_AK3Calo"),
                                             connect = cms.untracked.string("sqlite_file:/afs/cern.ch/user/j/jrobles/scratch0/newCMSSW_5_3_8_patch2/src/DB_JEC/v9/JEC_PP2760GEV_CMSSW538_2013.db"),
                                             label = cms.untracked.string("AK3Calo_HI")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             #                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_538_v07_AK4Calo_offline"),
                                             #                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_PythiaZ2_2760GeV_538_AK5Calo"),
                                             connect = cms.untracked.string("sqlite_file:/afs/cern.ch/user/j/jrobles/scratch0/newCMSSW_5_3_8_patch2/src/DB_JEC/v9/JEC_PP2760GEV_CMSSW538_2013.db"),
                                             label = cms.untracked.string("AK4Calo_HI")
                                             ),
                                    
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             #                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_538_v07_AK5Calo_offline"),
                                             #                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_PythiaZ2_2760GeV_538_AK5Calo"),
                                             connect = cms.untracked.string("sqlite_file:/afs/cern.ch/user/j/jrobles/scratch0/newCMSSW_5_3_8_patch2/src/DB_JEC/v9/JEC_PP2760GEV_CMSSW538_2013.db"),
                                             label = cms.untracked.string("AK5Calo_HI")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_5TeV_538_v04_AK6Calo_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK6Calo_HI")
                                             ),
                                    
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_5TeV_538_v03_AK1Calo_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AKPu1Calo_HI")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_5TeV_538_v04_AK2Calo_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AKPu2Calo_HI")
                                             ),
                                    
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             #                                           tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_538_v07_AK3Calo_offline"),
                                             #                                           connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_PythiaZ2_2760GeV_538_AK3Calo"),
                                             connect = cms.untracked.string("sqlite_file:/afs/cern.ch/user/j/jrobles/scratch0/newCMSSW_5_3_8_patch2/src/DB_JEC/v9/JEC_PP2760GEV_CMSSW538_2013.db"),
                                             label = cms.untracked.string("AKPu3Calo_HI")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             #                                           tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_538_v07_AK4Calo_offline"),
                                             #                                           connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_PythiaZ2_2760GeV_538_AK4Calo"),
                                             connect = cms.untracked.string("sqlite_file:/afs/cern.ch/user/j/jrobles/scratch0/newCMSSW_5_3_8_patch2/src/DB_JEC/v9/JEC_PP2760GEV_CMSSW538_2013.db"),
                                             label = cms.untracked.string("AKPu4Calo_HI")
                                             ),
                                    
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             #                                          tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_538_v07_AK5Calo_offline"),
                                             #                                          connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_PythiaZ2_2760GeV_538_AK5Calo"),
                                             connect = cms.untracked.string("sqlite_file:/afs/cern.ch/user/j/jrobles/scratch0/newCMSSW_5_3_8_patch2/src/DB_JEC/v9/JEC_PP2760GEV_CMSSW538_2013.db"),
                                             label = cms.untracked.string("AKPu5Calo_HI")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_5TeV_538_v04_AK6Calo_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AKPu6Calo_HI")
                                             ),
                                    
                                    
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v02_AK2PF_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK1PF_generalTracks")
                                             ),
                                    
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v04_AK2PF_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK2PF_generalTracks")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             #                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_2760GeV_538_v07_AK3PF_offline"),
                                             #                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_2760GeV_538_AK3PF"),
                                             connect = cms.untracked.string("sqlite_file:/afs/cern.ch/user/j/jrobles/scratch0/newCMSSW_5_3_8_patch2/src/DB_JEC/v9/JEC_PP2760GEV_CMSSW538_2013.db"),
                                             label = cms.untracked.string("AK3PF_generalTracks")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             #                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_2760GeV_538_v07_AK4PF_offline"),
                                             #                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_2760GeV_538_AK4PF"),
                                             connect = cms.untracked.string("sqlite_file:/afs/cern.ch/user/j/jrobles/scratch0/newCMSSW_5_3_8_patch2/src/DB_JEC/v9/JEC_PP2760GEV_CMSSW538_2013.db"),
                                             label = cms.untracked.string("AK4PF_generalTracks")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             #                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_2760GeV_538_v07_AK5PF_offline"),
                                             #                 connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_2760GeV_538_AK5PF"),
                                             connect = cms.untracked.string("sqlite_file:/afs/cern.ch/user/j/jrobles/scratch0/newCMSSW_5_3_8_patch2/src/DB_JEC/v9/JEC_PP2760GEV_CMSSW538_2013.db"),
                                             label = cms.untracked.string("AK5PF_generalTracks")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v04_AK6PF_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK6PF_generalTracks")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_Fall12_V5_DATA_AK7PF"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK7PF_generalTracks")
                                             ),
                                    
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v02_AK2PF_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AKPu1PF_generalTracks")
                                             ),
                                    
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v04_AK2PF_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AKPu2PF_generalTracks")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             #                                           tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_2760GeV_538_v07_AK3PF_offline"),
                                             #                                           connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_2760GeV_538_AK3PF"),
                                             connect = cms.untracked.string("sqlite_file:/afs/cern.ch/user/j/jrobles/scratch0/newCMSSW_5_3_8_patch2/src/DB_JEC/v9/JEC_PP2760GEV_CMSSW538_2013.db"),
                                             
                                             label = cms.untracked.string("AKPu3PF_generalTracks")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             #                                          tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_2760GeV_538_v07_AK4PF_offline"),
                                             #                                          connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_2760GeV_538_AK4PF"),
                                             connect = cms.untracked.string("sqlite_file:/afs/cern.ch/user/j/jrobles/scratch0/newCMSSW_5_3_8_patch2/src/DB_JEC/v9/JEC_PP2760GEV_CMSSW538_2013.db"),
                                             
                                             label = cms.untracked.string("AKPu4PF_generalTracks")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             #                                           tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_2760GeV_538_v07_AK5PF_offline"),
                                             #                                           connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_2760GeV_538_AK5PF"),
                                             connect = cms.untracked.string("sqlite_file:/afs/cern.ch/user/j/jrobles/scratch0/newCMSSW_5_3_8_patch2/src/DB_JEC/v9/JEC_PP2760GEV_CMSSW538_2013.db"),
                                             
                                             label = cms.untracked.string("AKPu5PF_generalTracks")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v04_AK6PF_offline"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AKPu6PF_generalTracks")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                             tag = cms.string("JetCorrectorParametersCollection_Fall12_V5_DATA_AK7PF"),
                                             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AKPu7PF_generalTracks")
                                             ),
                                    
                                    ])
    
    return process

#======  Final default common functions including centrality

def overrideGT_pPb5020(process):
    overrideCentrality(process)
    overrideJEC_pPb5020(process)
    return process

def overrideGT_pp2760(process):
    overrideCentrality(process)
    overrideJEC_pp2760(process)
    return process

def overrideGT_PbPb2760(process):
    overrideCentrality(process)
    overrideJEC_PbPb2760(process)
    return process


