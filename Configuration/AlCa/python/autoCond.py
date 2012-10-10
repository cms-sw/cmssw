autoCond = { 
    # GlobalTag for MC production with perfectly aligned and calibrated detector
    'mc'                :   'MC_61_V1::All',
    # GlobalTag for MC production with realistic alignment and calibrations
    'startup'           :   'START61_V1::All',
    # GlobalTag for MC production of Heavy Ions events with realistic alignment and calibrations
    'starthi'           :   'STARTHI61_V3::All',
    # GlobalTag for data reprocessing: this should always be the GR_R tag
    'com10'             : ( 'GR_R_60_V7::All',
                            # HLT particle flow jet energy corrections
                            'JetCorrectorParametersCollection_AK5PF_2012_V8_hlt_mc,JetCorrectionsRecord,frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS,AK5PFHLT',
                            'JetCorrectorParametersCollection_AK5PFchs_2012_V8_hlt_mc,JetCorrectionsRecord,frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS,AK5PFchsHLT',
                          ),
    # GlobalTag for running HLT on recent data: this should be the GR_P (prompt reco) global tag until a compatible GR_H tag is available, 
    # then it should point to the GR_H tag and override the connection string and pfnPrefix for use offline
    'hltonline'         : ( 'GR_P_V42::All',
                            # HLT particle flow jet energy corrections
                            'JetCorrectorParametersCollection_AK5PF_2012_V8_hlt_mc,JetCorrectionsRecord,frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS,AK5PFHLT',
                            'JetCorrectorParametersCollection_AK5PFchs_2012_V8_hlt_mc,JetCorrectionsRecord,frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS,AK5PFchsHLT',
                          )
}

# dedicated GlobalTags for MC production with the frozen HLT menus
autoCond['startup_5E33v4'] = ( autoCond['startup'], ) + (
                            # L1 menu 2012 v0
                            'L1GtTriggerMenu_L1Menu_Collisions2012_v0_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                            # old L1 CSCTF configuration
                            'L1MuCSCPtLut_key-10_mc,L1MuCSCPtLutRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                            # L1 DTTF settings used up to Run2012B
                            'L1MuDTTFParameters_dttf11_TSC_09_17_col_mc,L1MuDTTFParametersRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                            # old L1 GCT configuration, without jet seed threshold
                            'L1GctJetFinderParams_GCTPhysics_2011_09_01_B_mc,L1GctJetFinderParamsRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                            'L1HfRingEtScale_GCTPhysics_2011_09_01_B_mc,L1HfRingEtScaleRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                            'L1HtMissScale_GCTPhysics_2011_09_01_B_mc,L1HtMissScaleRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                            'L1JetEtScale_GCTPhysics_2011_09_01_B_mc,L1JetEtScaleRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                          )
autoCond['startup_7E33v2'] = ( autoCond['startup'], ) + (
                            # L1 menu 2012 v1
                            'L1GtTriggerMenu_L1Menu_Collisions2012_v1a_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                            # L1 DTTF settings used up to Run2012B
                            'L1MuDTTFParameters_dttf11_TSC_09_17_col_mc,L1MuDTTFParametersRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                            # old L1 GCT configuration, without jet seed threshold
                            'L1GctJetFinderParams_GCTPhysics_2011_09_01_B_mc,L1GctJetFinderParamsRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                            'L1HfRingEtScale_GCTPhysics_2011_09_01_B_mc,L1HfRingEtScaleRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                            'L1HtMissScale_GCTPhysics_2011_09_01_B_mc,L1HtMissScaleRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                            'L1JetEtScale_GCTPhysics_2011_09_01_B_mc,L1JetEtScaleRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                          )
autoCond['startup_7E33v3'] = ( autoCond['startup'],
                            # L1 menu 2012 v2
                            'L1GtTriggerMenu_L1Menu_Collisions2012_v2_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                            # L1 DTTF settings used since Run2012C
                            'L1MuDTTFParameters_dttf12_TSC_03_csc_col_mc,L1MuDTTFParametersRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                          )
autoCond['startup_7E33v4'] = ( autoCond['startup'], ) + (
                            # L1 menu 2012 v2
                            'L1GtTriggerMenu_L1Menu_Collisions2012_v2_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                            # L1 DTTF settings used since Run2012C
                            'L1MuDTTFParameters_dttf12_TSC_03_csc_col_mc,L1MuDTTFParametersRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                          )
autoCond['startup_8E33v1'] = ( autoCond['startup'], ) + (
                            # L1 menu 2012 v3
                            'L1GtTriggerMenu_L1Menu_Collisions2012_v3_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                            # L1 DTTF settings used since Run2012C
                            'L1MuDTTFParameters_dttf12_TSC_03_csc_col_mc,L1MuDTTFParametersRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                          )
autoCond['startup_GRun'] = ( autoCond['startup'], ) + (
                            # L1 menu 2012 v3
                            'L1GtTriggerMenu_L1Menu_Collisions2012_v3_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                            # L1 DTTF settings used since Run2012C
                            'L1MuDTTFParameters_dttf12_TSC_03_csc_col_mc,L1MuDTTFParametersRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                          )
autoCond['starthi_HIon'] = ( autoCond['starthi'], ) + (
                            # L1 heavy ions menu 2011 v0
                            'L1GtTriggerMenu_L1Menu_CollisionsHeavyIons2011_v0_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                          )

# dedicated GlobalTags for running the frozen HLT menus on data
autoCond['hltonline_5E33v4'] = autoCond['hltonline'] + (
                            # L1 menu 2012 v0
                            'L1GtTriggerMenu_L1Menu_Collisions2012_v0_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                          )
autoCond['hltonline_7E33v2'] = autoCond['hltonline'] + (
                            # L1 menu 2012 v1
                            'L1GtTriggerMenu_L1Menu_Collisions2012_v1a_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                          )
autoCond['hltonline_7E33v3'] = autoCond['hltonline'] + (
                            # L1 menu 2012 v2
                            'L1GtTriggerMenu_L1Menu_Collisions2012_v2_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                          )
autoCond['hltonline_7E33v4'] = autoCond['hltonline'] + (
                            # L1 menu 2012 v2
                            'L1GtTriggerMenu_L1Menu_Collisions2012_v2_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                          )
autoCond['hltonline_8E33v1'] = autoCond['hltonline'] + (
                            # L1 menu 2012 v3
                            'L1GtTriggerMenu_L1Menu_Collisions2012_v3_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                          )
autoCond['hltonline_GRun'] = autoCond['hltonline'] + (
                            # L1 menu 2012 v3
                            'L1GtTriggerMenu_L1Menu_Collisions2012_v3_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                          )
autoCond['hltonline_HIon'] = autoCond['hltonline'] + (
                            # L1 heavy ions menu 2011 v0
                            'L1GtTriggerMenu_L1Menu_CollisionsHeavyIons2011_v0_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                          )

# dedicated GlobalTags for running RECO and the frozen HLT menus on data
autoCond['com10_5E33v4'] = autoCond['com10'] + (
                            # L1 menu 2012 v0
                            'L1GtTriggerMenu_L1Menu_Collisions2012_v0_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                          )
autoCond['com10_7E33v2'] = autoCond['com10'] + (
                            # L1 menu 2012 v1
                            'L1GtTriggerMenu_L1Menu_Collisions2012_v1a_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                          )
autoCond['com10_7E33v3'] = autoCond['com10'] + (
                            # L1 menu 2012 v2
                            'L1GtTriggerMenu_L1Menu_Collisions2012_v2_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                          )
autoCond['com10_7E33v4'] = autoCond['com10'] + (
                            # L1 menu 2012 v2
                            'L1GtTriggerMenu_L1Menu_Collisions2012_v2_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                          )
autoCond['com10_8E33v1'] = autoCond['com10'] + (
                            # L1 menu 2012 v3
                            'L1GtTriggerMenu_L1Menu_Collisions2012_v3_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                          )
autoCond['com10_GRun'] = autoCond['com10'] + (
                            # L1 menu 2012 v3
                            'L1GtTriggerMenu_L1Menu_Collisions2012_v3_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                          )
autoCond['com10_HIon'] = autoCond['com10'] + (
                            # L1 heavy ions menu 2011 v0
                            'L1GtTriggerMenu_L1Menu_CollisionsHeavyIons2011_v0_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                          )

