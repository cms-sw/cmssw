autoCond = { 
    'upgrade2017'       :   'DES17_62_V8::All', # 
    'upgrade2019'       :   'DES19_62_V8::All', # 
    'upgradePLS3'       :   'DES23_62_V1::All' # 
}

aliases = {
    'MAINGT' : 'FT_P_V42D::All|AN_V4::All',
    'BASEGT' : 'BASE1_V1::All|BASE2_V1::All'
}

# L1 configuration used during Run2012A
conditions_L1_Run2012A = (
    # L1 GT menu 2012 v0, used during Run2012A
    'L1GtTriggerMenu_L1Menu_Collisions2012_v0_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
    # L1 GCT configuration without jet seed threshold, used up to Run2012B
    'L1GctJetFinderParams_GCTPhysics_2011_09_01_B_mc,L1GctJetFinderParamsRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
    'L1HfRingEtScale_GCTPhysics_2011_09_01_B_mc,L1HfRingEtScaleRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
    'L1HtMissScale_GCTPhysics_2011_09_01_B_mc,L1HtMissScaleRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
    'L1JetEtScale_GCTPhysics_2011_09_01_B_mc,L1JetEtScaleRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
    # L1 CSCTF configuration used up to Run2012A
    'L1MuCSCPtLut_key-10_mc,L1MuCSCPtLutRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
    # L1 DTTF settings used up to Run2012B
    'L1MuDTTFParameters_dttf11_TSC_09_17_col_mc,L1MuDTTFParametersRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
)

# L1 configuration used during Run2012B
conditions_L1_Run2012B = (
    # L1 GT menu 2012 v1, used during Run2012B
    'L1GtTriggerMenu_L1Menu_Collisions2012_v1a_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
    # L1 GCT configuration without jet seed threshold, used up to Run2012B
    'L1GctJetFinderParams_GCTPhysics_2011_09_01_B_mc,L1GctJetFinderParamsRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
    'L1HfRingEtScale_GCTPhysics_2011_09_01_B_mc,L1HfRingEtScaleRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
    'L1HtMissScale_GCTPhysics_2011_09_01_B_mc,L1HtMissScaleRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
    'L1JetEtScale_GCTPhysics_2011_09_01_B_mc,L1JetEtScaleRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
    # L1 CSCTF configuration used since Run2012B
    'L1MuCSCPtLut_key-11_mc,L1MuCSCPtLutRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
    # L1 DTTF settings used up to Run2012B
    'L1MuDTTFParameters_dttf11_TSC_09_17_col_mc,L1MuDTTFParametersRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
)

# L1 configuration used during Run2012C
conditions_L1_Run2012C = (
    # L1 GT menu 2012 v2, used during Run2012C
    'L1GtTriggerMenu_L1Menu_Collisions2012_v2_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
    # L1 GCT configuration with 5 GeV jet seed threshold, used since Run2012C
    'L1GctJetFinderParams_GCTPhysics_2012_04_27_JetSeedThresh5GeV_mc,L1GctJetFinderParamsRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
    'L1HfRingEtScale_GCTPhysics_2012_04_27_JetSeedThresh5GeV_mc,L1HfRingEtScaleRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
    'L1HtMissScale_GCTPhysics_2012_04_27_JetSeedThresh5GeV_mc,L1HtMissScaleRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
    'L1JetEtScale_GCTPhysics_2012_04_27_JetSeedThresh5GeV_mc,L1JetEtScaleRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
    # L1 CSCTF configuration used since Run2012B
    'L1MuCSCPtLut_key-11_mc,L1MuCSCPtLutRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
    # L1 DTTF settings used since Run2012C
    'L1MuDTTFParameters_dttf12_TSC_03_csc_col_mc,L1MuDTTFParametersRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
)


# L1 configuration used during Run2012D
conditions_L1_Run2012D = (
    # L1 GT menu 2012 v3, used during Run2012D
    'L1GtTriggerMenu_L1Menu_Collisions2012_v3_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
    # L1 GCT configuration with 5 GeV jet seed threshold, used since Run2012C
    'L1GctJetFinderParams_GCTPhysics_2012_04_27_JetSeedThresh5GeV_mc,L1GctJetFinderParamsRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
    'L1HfRingEtScale_GCTPhysics_2012_04_27_JetSeedThresh5GeV_mc,L1HfRingEtScaleRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
    'L1HtMissScale_GCTPhysics_2012_04_27_JetSeedThresh5GeV_mc,L1HtMissScaleRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
    'L1JetEtScale_GCTPhysics_2012_04_27_JetSeedThresh5GeV_mc,L1JetEtScaleRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
    # L1 CSCTF configuration used since Run2012B
    'L1MuCSCPtLut_key-11_mc,L1MuCSCPtLutRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
    # L1 DTTF settings used since Run2012C
    'L1MuDTTFParameters_dttf12_TSC_03_csc_col_mc,L1MuDTTFParametersRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
)

# L1 configuration used during HIRun2011
conditions_L1_HIRun2011 = (
    # L1 heavy ions menu 2011 v0
    'L1GtTriggerMenu_L1Menu_CollisionsHeavyIons2011_v0_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
)

# L1 configuration used during PARun2013
conditions_L1_PARun2013 = (
    # L1 GT menu HI 2013 v0, used for the p-Pb run 2013
    'L1GtTriggerMenu_L1Menu_CollisionsHeavyIons2013_v0_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
    # L1 GCT configuration without jet seed threshold (same as 2012B)
    'L1GctJetFinderParams_GCTPhysics_2011_09_01_B_mc,L1GctJetFinderParamsRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
    'L1HfRingEtScale_GCTPhysics_2011_09_01_B_mc,L1HfRingEtScaleRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
    'L1HtMissScale_GCTPhysics_2011_09_01_B_mc,L1HtMissScaleRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
    'L1JetEtScale_GCTPhysics_2011_09_01_B_mc,L1JetEtScaleRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
    # L1 CSCTF configuration used since Run2012B
    'L1MuCSCPtLut_key-11_mc,L1MuCSCPtLutRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
    # L1 DTTF settings used since Run2012C
    'L1MuDTTFParameters_dttf12_TSC_03_csc_col_mc,L1MuDTTFParametersRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
)

# HLT Jet Energy Corrections
conditions_HLT_JECs = (
    # HLT 2012 jet energy corrections
    'JetCorrectorParametersCollection_Jec11_V12_AK5CaloHLT,JetCorrectionsRecord,frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS,AK5CaloHLT',
    'JetCorrectorParametersCollection_AK5PF_2012_V8_hlt_mc,JetCorrectionsRecord,frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS,AK5PFHLT',
    'JetCorrectorParametersCollection_AK5PFchs_2012_V8_hlt_mc,JetCorrectionsRecord,frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS,AK5PFchsHLT',
)


# dedicated GlobalTags for MC production with the frozen HLT menus
