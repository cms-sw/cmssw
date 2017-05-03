
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
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_BEAMSPOT")
             ),
    )
    return process


def addRPFlat(process):
    process.GlobalTag.toGet.extend([
        cms.PSet(record = cms.string("HeavyIonRPRcd"),
                 tag = cms.string("RPFlatParams_Test_v0_offline"),
                 connect = cms.string("frontier://FrontierPrep/CMS_COND_TEMP"),
             ),
    ])
    return process


def overrideCentrality(process):
    process.GlobalTag.toGet.extend([

#==================== MC Tables ====================
        cms.PSet(record = cms.string("HeavyIonRcd"),
                 tag = cms.string("CentralityTable_HFhits40_AMPTOrgan_v0_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("HFhitsAMPT_Organ")
             ),
        cms.PSet(record = cms.string("HeavyIonRcd"),
                 tag = cms.string("CentralityTable_PixelHits40_AMPTOrgan_v0_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("PixelHitsAMPT_Organ")
             ),

        cms.PSet(record = cms.string("HeavyIonRcd"),
                 tag = cms.string("CentralityTable_HFhits40_HydjetBass_vv44x04_mc"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("HFhitsHydjet_Bass")
             ),
        cms.PSet(record = cms.string("HeavyIonRcd"),
                 tag = cms.string("CentralityTable_PixelHits40_HydjetBass_vv44x04_mc"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("PixelHitsHydjet_Bass")
             ),
        cms.PSet(record = cms.string("HeavyIonRcd"),
                 tag = cms.string("CentralityTable_Tracks40_HydjetBass_vv44x04_mc"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("TracksHydjet_Bass")
             ),
        cms.PSet(record = cms.string("HeavyIonRcd"),
                 tag = cms.string("CentralityTable_PixelTracks40_HydjetBass_vv44x04_mc"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("PixelTracksHydjet_Bass")
             ),

        cms.PSet(record = cms.string("HeavyIonRcd"),
                 tag = cms.string("CentralityTable_HFtowers40_HydjetBass_vv44x04_mc"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("HFtowersHydjet_Bass")
             ),

        cms.PSet(record = cms.string("HeavyIonRcd"),
                 tag = cms.string("CentralityTable_HFhits40_HydjetDrum_vv44x05_mc"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("HFhitsHydjet_Drum")
             ),
        cms.PSet(record = cms.string("HeavyIonRcd"),
                 tag = cms.string("CentralityTable_PixelHits40_HydjetDrum_vv44x05_mc"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("PixelHitsHydjet_Drum")
             ),
        cms.PSet(record = cms.string("HeavyIonRcd"),
                 tag = cms.string("CentralityTable_Tracks40_HydjetDrum_vv44x05_mc"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("TracksHydjet_Drum")
             ),
        cms.PSet(record = cms.string("HeavyIonRcd"),
                 tag = cms.string("CentralityTable_PixelTracks40_HydjetDrum_vv44x05_mc"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("PixelTracksHydjet_Drum")
             ),

        cms.PSet(record = cms.string("HeavyIonRcd"),
                 tag = cms.string("CentralityTable_HFtowers200_HydjetDrum_v5315x01_mc"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("HFtowersHydjet_Drum")
             ),
        cms.PSet(record = cms.string("HeavyIonRcd"),
                 tag = cms.string("CentralityTable_PixelHits40_Glauber2010A_v3_effB_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("PixelHits")
             ),

        cms.PSet(record = cms.string("HeavyIonRcd"),
                 tag = cms.string("CentralityTable_HFhits40_Glauber2010A_v3_effB_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("HFhits")
             ),

        cms.PSet(record = cms.string("HeavyIonRcd"),
                 tag = cms.string("CentralityTable_HFtowers200_Glauber2010A_v5315x01_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("HFtowers")
             ),

#==================== pPb data taking 2013 =====================================

        cms.PSet(record = cms.string("HeavyIonRcd"),
                 tag = cms.string("CentralityTable_HFplus100_PA2012B_v533x01_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("HFtowersPlusTrunc")
             ),
        cms.PSet(record = cms.string("HeavyIonRcd"),
                 tag = cms.string("CentralityTable_HFtrunc100_PA2012B_v538x02_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("HFtowersTrunc")
             ),

#==================== pPb MC 2013 =====================================

        cms.PSet(record = cms.string("HeavyIonRcd"),
                 tag = cms.string("CentralityTable_Tracks100_HijingPA_v538x02_mc"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("TracksHijing")
             ),
        cms.PSet(record = cms.string("HeavyIonRcd"),
                 tag = cms.string("CentralityTable_HFplus100_HijingPA_v538x02_mc"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("HFtowersPlusTruncHijing")
             ),
        cms.PSet(record = cms.string("HeavyIonRcd"),
                 tag = cms.string("CentralityTable_HFtrunc100_HijingPA_v538x01_mc"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("HFtowersTruncHijing")
             ),
        cms.PSet(record = cms.string("HeavyIonRcd"),
                 tag = cms.string("CentralityTable_HFplus100_EposPA_v538x01_mc"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("HFtowersPlusTruncEpos")
             ),
        cms.PSet(record = cms.string("HeavyIonRcd"),
                 tag = cms.string("CentralityTable_HFtrunc100_EposPA_v538x01_mc"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("HFtowersTruncEpos")
             ),

    ])
    return process

def overrideJEC_PbPb2760(process):
    process.GlobalTag.toGet.extend([
## no PU or VS
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AK1Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK1Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AK2Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK2Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AK3Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK3Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AK4Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK4Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AK5Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK5Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AK6Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK6Calo_HI")
             ),
         cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AK7Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK7Calo_HI")
             ),            
             
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AK1PF_offline"),
                                             connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK1PF_hiIterativeTracks")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AK2PF_offline"),
                                             connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK2PF_hiIterativeTracks")
                                             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK3PF_hiIterativeTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AK4PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK4PF_hiIterativeTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AK5PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK5PF_hiIterativeTracks")
             ),
                                    cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AK6PF_offline"),
                                             connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK6PF_hiIterativeTracks")
                                             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AK7PF_offline"),
                                             connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK7PF_hiIterativeTracks")
                                             ),
## Pu
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKPu1Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu1Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKPu2Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu2Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKPu3Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu3Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKPu4Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu4Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKPu5Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu5Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKPu6Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu6Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKPu7Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu7Calo_HI")
             ),             
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKPu1PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu1PF_hiIterativeTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKPu2PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu2PF_hiIterativeTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKPu3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu3PF_hiIterativeTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKPu4PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu4PF_hiIterativeTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKPu5PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu5PF_hiIterativeTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKPu6PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu6PF_hiIterativeTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKPu7PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu7PF_hiIterativeTracks")
             ),
## VS
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKVs1Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs1Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKVs2Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs2Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKVs3Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs3Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKVs4Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs4Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKVs5Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs5Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKVs6Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs6Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKVs7Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs7Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKVs1PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs1PF_hiIterativeTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKVs2PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs2PF_hiIterativeTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKVs3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs3PF_hiIterativeTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKVs4PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs4PF_hiIterativeTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKVs5PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs5PF_hiIterativeTracks")
                                             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKVs6PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs6PF_hiIterativeTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKVs7PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs7PF_hiIterativeTracks")
                                             ),                                             
## generalTracks
## no PU or VS
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AK1PF_offline"),
                                             connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK1PF_generalTracks")
                                             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AK2PF_offline"),
                                             connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK2PF_generalTracks")
                                             ),                               
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AK3PF_offline"),
                                             connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK3PF_generalTracks")
                                             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AK4PF_offline"),
                                             connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK4PF_generalTracks")
                                             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AK5PF_offline"),
                                             connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK5PF_generalTracks")
                                             ),                                                                                 
     	cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AK6PF_offline"),
                                             connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK6PF_generalTracks")
                                             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AK7PF_offline"),
                                             connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                             label = cms.untracked.string("AK7PF_generalTracks")
                                             ),
## Pu      
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKPu1PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu1PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKPu2PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu2PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKPu3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu3PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKPu4PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu4PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKPu5PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu5PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKPu6PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu6PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKPu7PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu7PF_generalTracks")
             ),
## VS
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKVs1PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs1PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKVs2PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs2PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKVs3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs3PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKVs4PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs4PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKVs5PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs5PF_generalTracks")
                                             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKVs6PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs6PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v15_AKVs7PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs7PF_generalTracks")
                                             ),      
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_IC5Calo_2760GeV_v0_offline"),
                 # pp 7TeV version:       JetCorrectorParametersCollection_Fall12_V5_DATA_IC5Calo
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("IC5Calo_2760GeV")
             ),
         cms.PSet(record = cms.string("JetCorrectionsRecord"),
                # tag = cms.string("JetCorrectorParametersCollection_Summer13_V4_MC_AK5JPT"),
                 tag = cms.string("JetCorrectorParametersCollection_Fall12_V5_MC_AK5JPT"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK5JPT")
                 ),
    ])
    return process


def overrideJEC_pPb5020(process):
    process.GlobalTag.toGet.extend([
## no PU
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_5TeV_538_v03_AK1Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK1Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_5TeV_538_v04_AK2Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK2Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_5TeV_538_v05_AK3Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK3Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_5TeV_538_v05_AK4Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK4Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_5TeV_538_v05_AK5Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK5Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_5TeV_538_v04_AK6Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK6Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v04_AK2PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK2PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK3PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK4PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK4PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK5PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK5PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v04_AK6PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK6PF_generalTracks")
             ),
## Pu
       cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_5TeV_538_v08_AKPu3Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu3Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_5TeV_538_v08_AKPu4Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu4Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_5TeV_538_v08_AKPu5Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu5Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v08_AKPu3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu3PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v08_AKPu4PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu4PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v08_AKPu5PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu5PF_generalTracks")
             ),
    ])
    return process

def overrideJEC_Pbp5020(process):
    process.GlobalTag.toGet.extend([
## no Pu
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_ppTracking_Pbp_PythiaZ2_5020GeV_538HIp2_v17_AK3Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK3Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_ppTracking_Pbp_PythiaZ2_5020GeV_538HIp2_v17_AK4Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK4Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_ppTracking_Pbp_PythiaZ2_5020GeV_538HIp2_v17_AK5Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK5Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_ppTracking_Pbp_PythiaZ2_5020GeV_538HIp2_v17_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK3PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_ppTracking_Pbp_PythiaZ2_5020GeV_538HIp2_v17_AK4PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK4PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_ppTracking_Pbp_PythiaZ2_5020GeV_538HIp2_v17_AK5PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK5PF_generalTracks")
             ),
## Pu
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_ppTracking_Pbp_PythiaZ2_5020GeV_538HIp2_v17_AKPu3Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu3Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_ppTracking_Pbp_PythiaZ2_5020GeV_538HIp2_v17_AKPu4Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu4Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_ppTracking_Pbp_PythiaZ2_5020GeV_538HIp2_v17_AKPu5Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu5Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_ppTracking_Pbp_PythiaZ2_5020GeV_538HIp2_v17_AKPu3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu3PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_ppTracking_Pbp_PythiaZ2_5020GeV_538HIp2_v17_AKPu4PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu4PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_ppTracking_Pbp_PythiaZ2_5020GeV_538HIp2_v17_AKPu5PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu5PF_generalTracks")
             ),
    ])
    return process

def overrideJEC_MC_Pbp5020(process):
    process.GlobalTag.toGet.extend([
        ## no Pu
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
            tag = cms.string("JetCorrectorParametersCollection_ppTracking_Pbp_PythiaZ2_5020GeV_538HIp2_v19_reversed_AK3Calo_mc"),
            connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
            label = cms.untracked.string("AK3Calo_HI")
            ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
            tag = cms.string("JetCorrectorParametersCollection_ppTracking_Pbp_PythiaZ2_5020GeV_538HIp2_v19_reversed_AK4Calo_mc"),
            connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
            label = cms.untracked.string("AK4Calo_HI")
            ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
            tag = cms.string("JetCorrectorParametersCollection_ppTracking_Pbp_PythiaZ2_5020GeV_538HIp2_v19_reversed_AK5Calo_mc"),
            connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
            label = cms.untracked.string("AK5Calo_HI")
            ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
            tag = cms.string("JetCorrectorParametersCollection_ppTracking_Pbp_PythiaZ2_5020GeV_538HIp2_v19_reversed_AK3PF_mc"),
            connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
            label = cms.untracked.string("AK3PF_generalTracks")
            ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
            tag = cms.string("JetCorrectorParametersCollection_ppTracking_Pbp_PythiaZ2_5020GeV_538HIp2_v19_reversed_AK4PF_mc"),
            connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
            label = cms.untracked.string("AK4PF_generalTracks")
            ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
            tag = cms.string("JetCorrectorParametersCollection_ppTracking_Pbp_PythiaZ2_5020GeV_538HIp2_v19_reversed_AK5PF_mc"),
            connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
            label = cms.untracked.string("AK5PF_generalTracks")
            ),
        ## Pu
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
            tag = cms.string("JetCorrectorParametersCollection_ppTracking_Pbp_PythiaZ2_5020GeV_538HIp2_v19_reversed_AKPu3Calo_mc"),
            connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
            label = cms.untracked.string("AKPu3Calo_HI")
            ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
            tag = cms.string("JetCorrectorParametersCollection_ppTracking_Pbp_PythiaZ2_5020GeV_538HIp2_v19_reversed_AKPu4Calo_mc"),
            connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
            label = cms.untracked.string("AKPu4Calo_HI")
            ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
            tag = cms.string("JetCorrectorParametersCollection_ppTracking_Pbp_PythiaZ2_5020GeV_538HIp2_v19_reversed_AKPu5Calo_mc"),
            connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
            label = cms.untracked.string("AKPu5Calo_HI")
            ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
            tag = cms.string("JetCorrectorParametersCollection_ppTracking_Pbp_PythiaZ2_5020GeV_538HIp2_v19_reversed_AKPu3PF_mc"),
            connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
            label = cms.untracked.string("AKPu3PF_generalTracks")
            ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                tag = cms.string("JetCorrectorParametersCollection_ppTracking_Pbp_PythiaZ2_5020GeV_538HIp2_v19_reversed_AKPu4PF_mc"),
                connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                label = cms.untracked.string("AKPu4PF_generalTracks")
                ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                tag = cms.string("JetCorrectorParametersCollection_ppTracking_Pbp_PythiaZ2_5020GeV_538HIp2_v19_reversed_AKPu5PF_mc"),
                connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                label = cms.untracked.string("AKPu5PF_generalTracks")
                ),
        ])
    return process


def overrideJEC_pp5020(process):
    process.load("CondCore.DBCommon.CondDBCommon_cfi")
    #from CondCore.DBCommon.CondDBSetup_cfi import *
    process.jec = cms.ESSource("PoolDBESSource",
                               DBParameters = cms.PSet(
                                   messageLevel = cms.untracked.int32(0)
                               ),
                               timetype = cms.string('runnumber'),
                               toGet = cms.VPSet(
                                   cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                            tag = cms.string("JetCorrectorParametersCollection_HI_PythiaCUETP8M1_5020GeV_753p1_v12_AK1Calo_offline"),
                                            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
                                            label = cms.untracked.string("AK1Calo_offline")
                                   ),
                                   
                                   cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                            tag = cms.string("JetCorrectorParametersCollection_HI_PythiaCUETP8M1_5020GeV_753p1_v12_AK1PF_offline"),
                                            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
                                            label = cms.untracked.string("AK1PF_offline")
                                   ),
                                   cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                            tag = cms.string("JetCorrectorParametersCollection_HI_PythiaCUETP8M1_5020GeV_753p1_v12_AK2Calo_offline"),
                                            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
                                            label = cms.untracked.string("AK2Calo_offline")
                                   ),
                                   
                                   cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                            tag = cms.string("JetCorrectorParametersCollection_HI_PythiaCUETP8M1_5020GeV_753p1_v12_AK2PF_offline"),
                                            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
                                            label = cms.untracked.string("AK2PF_offline")
                                   ),
                                   cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                            tag = cms.string("JetCorrectorParametersCollection_HI_PythiaCUETP8M1_5020GeV_753p1_v12_AK3Calo_offline"),
                                            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
                                            label = cms.untracked.string("AK3Calo_offline")
                                   ),
                                   
                                   cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                            tag = cms.string("JetCorrectorParametersCollection_HI_PythiaCUETP8M1_5020GeV_753p1_v12_AK3PF_offline"),
                                            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
                                            label = cms.untracked.string("AK3PF_offline")
                                   ),
                                   cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                            tag = cms.string("JetCorrectorParametersCollection_HI_PythiaCUETP8M1_5020GeV_753p1_v12_AK4Calo_offline"),
                                            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
                                            label = cms.untracked.string("AK4Calo_offline")
                                   ),
                                   
                                   cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                            tag = cms.string("JetCorrectorParametersCollection_HI_PythiaCUETP8M1_5020GeV_753p1_v7_AK4PF_offline"),
                                            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
                                            label = cms.untracked.string("AK4PF_offline")
                                   ),
                                   cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                            tag = cms.string("JetCorrectorParametersCollection_HI_PythiaCUETP8M1_5020GeV_753p1_v12_AK5Calo_offline"),
                                            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
                                            label = cms.untracked.string("AK5Calo_offline")
                                   ),
                                   
                                   cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                            tag = cms.string("JetCorrectorParametersCollection_HI_PythiaCUETP8M1_5020GeV_753p1_v12_AK5PF_offline"),
                                            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
                                            label = cms.untracked.string("AK5PF_offline")
                                   )
                                   ,
                                   cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                            tag = cms.string("JetCorrectorParametersCollection_HI_PythiaCUETP8M1_5020GeV_753p1_v12_AK6Calo_offline"),
                                            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
                                            label = cms.untracked.string("AK6Calo_offline")
                                   ),
                                   
                                   cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                            tag = cms.string("JetCorrectorParametersCollection_HI_PythiaCUETP8M1_5020GeV_753p1_v12_AK6PF_offline"),
                                            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
                                            label = cms.untracked.string("AK6PF_offline")
                                   )
                               )
    )
                               ## add an es_prefer statement to resolve a possible conflict from simultaneous connection to a global tag
    process.es_prefer_jec = cms.ESPrefer('PoolDBESSource','jec')
    return process

def overrideJEC_PbPb5020(process):
    #process.GlobalTag.toGet.extend([
    process.load("CondCore.DBCommon.CondDBCommon_cfi")
    #from CondCore.DBCommon.CondDBSetup_cfi import *
    process.jec = cms.ESSource("PoolDBESSource",
                               DBParameters = cms.PSet(
                                   messageLevel = cms.untracked.int32(0)
                               ),
                               timetype = cms.string('runnumber'),
                               toGet = cms.VPSet(
 
                                   cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                            tag = cms.string("JetCorrectorParametersCollection_HI_PythiaCUETP8M1_5020GeV_753p1_v10_AK1PF_offline"),
                                            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
                                            label = cms.untracked.string("AK1PF_offline")
                                   ),
                                   cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                            tag = cms.string("JetCorrectorParametersCollection_HI_PythiaCUETP8M1_5020GeV_753p1_v10_AK1Calo_offline"),
                                            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
                                            label = cms.untracked.string("AK1Calo_offline")
                                   ),
                                   cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                            tag = cms.string("JetCorrectorParametersCollection_HI_PythiaCUETP8M1_5020GeV_753p1_v10_AK2PF_offline"),
                                            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
                                            label = cms.untracked.string("AK2PF_offline")
                                   ),
                                   cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                            tag = cms.string("JetCorrectorParametersCollection_HI_PythiaCUETP8M1_5020GeV_753p1_v10_AK2Calo_offline"),
                 connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
                                            label = cms.untracked.string("AK2Calo_offline")
                                   ),
                                   cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                            tag = cms.string("JetCorrectorParametersCollection_HI_PythiaCUETP8M1_5020GeV_753p1_v10_AK3Calo_offline"),
                                            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
                                            label = cms.untracked.string("AK3Calo_offline")
                                   ),
                                   cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                            tag = cms.string("JetCorrectorParametersCollection_HI_PythiaCUETP8M1_5020GeV_753p1_v10_AK3PF_offline"),
                                            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
                                            label = cms.untracked.string("AK3PF_offline")
                                   ),

                                   cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                            tag = cms.string("JetCorrectorParametersCollection_HI_PythiaCUETP8M1_5020GeV_753p1_v10_AK4PF_offline"),
                                            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
                                        label = cms.untracked.string("AK4PF_offline")
                                   ),
                                   cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                            tag = cms.string("JetCorrectorParametersCollection_HI_PythiaCUETP8M1_5020GeV_753p1_v10_AK4Calo_offline"),
                                            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
                                            label = cms.untracked.string("AK4Calo_offline")
                                   ),
                                   cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                            tag = cms.string("JetCorrectorParametersCollection_HI_PythiaCUETP8M1_5020GeV_753p1_v10_AK5Calo_offline"),
                                            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
                                            label = cms.untracked.string("AK5Calo_offline")
                                   ),
                                   
                                   cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                            tag = cms.string("JetCorrectorParametersCollection_HI_PythiaCUETP8M1_5020GeV_753p1_v10_AK5PF_offline"),
                                            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
                                            label = cms.untracked.string("AK5PF_offline")
                                   ),
                                   cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                            tag = cms.string("JetCorrectorParametersCollection_HI_PythiaCUETP8M1_5020GeV_753p1_v10_AK6Calo_offline"),
                                            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
                                            label = cms.untracked.string("AK6Calo_offline")
                                   ),
                                   cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                            tag = cms.string("JetCorrectorParametersCollection_HI_PythiaCUETP8M1_5020GeV_753p1_v10_AK6PF_offline"),
                                            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
                                            label = cms.untracked.string("AK6PF_offline")
                                   )
    )
	)
                               ## add an es_prefer statement to resolve a possible conflict from simultaneous connection to a global tag
    process.es_prefer_jec = cms.ESPrefer('PoolDBESSource','jec')
    return process

def overrideJEC_pp2760(process):
    process.GlobalTag.toGet.extend([

        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_pp_PythiaZ2_2760GeV_538_v85_AK3Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK3Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v14_AK4Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK4Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v14_AK5Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK5Calo_HI")
             ),


        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_pp_PythiaZ2_2760GeV_538_v85_AK3Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu3Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v14_AKPu4Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu4Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v14_AKPu5Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu5Calo_HI")
             ),


        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_pp_PythiaZ2_2760GeV_538_v85_AK3Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs3Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v14_AKVs4Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs4Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaZ2_2760GeV_5316_v14_AKVs5Calo_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs5Calo_HI")
             ),



        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v02_AK2PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK1PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v04_AK2PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK2PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_2760GeV_538_v09_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK3PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_2760GeV_538_v09_AK4PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK4PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_2760GeV_538_v09_AK5PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK5PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v04_AK6PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK6PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_Fall12_V5_DATA_AK7PF"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK7PF_generalTracks")
             ),

        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v02_AK2PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu1PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v04_AK2PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu2PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_2760GeV_538_v09_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu3PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_2760GeV_538_v09_AK4PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu4PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_2760GeV_538_v09_AK5PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu5PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v04_AK6PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu6PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_Fall12_V5_DATA_AK7PF"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu7PF_generalTracks")
             ),

###PLACEHOLDERS BEGIN###

        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs3PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK4PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs4PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK5PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs5PF_generalTracks")
             ),


        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs2PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu2Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs2Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK2Calo_HI")
             ),

        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs1PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu1Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs1Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK1Calo_HI")
             ),


        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs6PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu6Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs6Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK6Calo_HI")
             ),


        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs7PF_generalTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu7Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs7Calo_HI")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AK7Calo_HI")
             ),

 # needed for pp MC RECO with PbPb 2011 conditions and use override_pp2760:
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu1PF_hiIterativeTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs1PF_hiIterativeTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs2PF_hiIterativeTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu2PF_hiIterativeTracks")
             ),

        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs3PF_hiIterativeTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu3PF_hiIterativeTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs4PF_hiIterativeTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu4PF_hiIterativeTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs5PF_hiIterativeTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu5PF_hiIterativeTracks")
             ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs6PF_hiIterativeTracks")
        ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu6PF_hiIterativeTracks")
        ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKVs7PF_hiIterativeTracks")
        ),
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK3PF_offline"),
                 connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                 label = cms.untracked.string("AKPu7PF_hiIterativeTracks")
        ),

###PLACEHOLDERS END###

    ])

    return process

## a set of intentionally-bad JEC for use when foresting all Jet
## algorithms and you don't care about a good JEC
def overrideJEC_NULL(process):
    allpayloads = ["AK1Calo_HI",
                   "AK2Calo_HI",
                   "AK3Calo_HI",
                   "AK4Calo_HI",
                   "AK5Calo_HI",
                   "AK6Calo_HI",
                   "AK7Calo_HI",
                   "AK1PF_generalTracks",
                   "AK2PF_generalTracks",
                   "AK3PF_generalTracks",
                   "AK4PF_generalTracks",
                   "AK5PF_generalTracks",
                   "AK6PF_generalTracks",
                   "AK7PF_generalTracks",
                   "AK1PF_hiIterativeTracks",
                   "AK2PF_hiIterativeTracks",
                   "AK3PF_hiIterativeTracks",
                   "AK4PF_hiIterativeTracks",
                   "AK5PF_hiIterativeTracks",
                   "AK6PF_hiIterativeTracks",
                   "AK7PF_hiIterativeTracks",
                   "AKPu1Calo_HI",
                   "AKPu2Calo_HI",
                   "AKPu3Calo_HI",
                   "AKPu4Calo_HI",
                   "AKPu5Calo_HI",
                   "AKPu6Calo_HI",
                   "AKPu7Calo_HI",
                   "AKPu1PF_generalTracks",
                   "AKPu2PF_generalTracks",
                   "AKPu3PF_generalTracks",
                   "AKPu4PF_generalTracks",
                   "AKPu5PF_generalTracks",
                   "AKPu6PF_generalTracks",
                   "AKPu7PF_generalTracks",
                   "AKPu1PF_hiIterativeTracks",
                   "AKPu2PF_hiIterativeTracks",
                   "AKPu3PF_hiIterativeTracks",
                   "AKPu4PF_hiIterativeTracks",
                   "AKPu5PF_hiIterativeTracks",
                   "AKPu6PF_hiIterativeTracks",
                   "AKPu7PF_hiIterativeTracks",
                   "AKVs1Calo_HI",
                   "AKVs2Calo_HI",
                   "AKVs3Calo_HI",
                   "AKVs4Calo_HI",
                   "AKVs5Calo_HI",
                   "AKVs6Calo_HI",
                   "AKVs7Calo_HI",
                   "AKVs1PF_generalTracks",
                   "AKVs2PF_generalTracks",
                   "AKVs3PF_generalTracks",
                   "AKVs4PF_generalTracks",
                   "AKVs5PF_generalTracks",
                   "AKVs6PF_generalTracks",
                   "AKVs7PF_generalTracks",
                   "AKVs1PF_hiIterativeTracks",
                   "AKVs2PF_hiIterativeTracks",
                   "AKVs3PF_hiIterativeTracks",
                   "AKVs4PF_hiIterativeTracks",
                   "AKVs5PF_hiIterativeTracks",
                   "AKVs6PF_hiIterativeTracks",
                   "AKVs7PF_hiIterativeTracks"]
    for payload in allpayloads:
        process.GlobalTag.toGet.extend([
            cms.PSet(record = cms.string("JetCorrectionsRecord"),
                     tag = cms.string("JetCorrectorParametersCollection_HI_PFTowers_generalTracks_PythiaZ2_5TeV_538_v05_AK3PF_offline"),
                     connect = cms.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                     label = cms.untracked.string(payload)
                 )
        ])
    return process




#======  Final default common functions including centrality

# Do not use the combo function for pPb, it causes too much confusion
# for users
#def overrideGT_pPb5020(process):
#    overrideCentrality(process)
#    overrideJEC_pPb5020(process)
#    return process

def overrideGT_pp2760(process):
    overrideCentrality(process)
    overrideJEC_pp2760(process)
    return process

def overrideGT_PbPb2760(process):
    overrideCentrality(process)
    overrideJEC_PbPb2760(process)
    return process
