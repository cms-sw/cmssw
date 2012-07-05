autoCond = {
    # global tag for MC production with ideal conditions 
    'mc'          : ( 'MC_52_V11::All',
                      # load the 2012 v2 L1 menu 
                      'L1GtTriggerMenu_L1Menu_Collisions2012_v2_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                      # load the 5 GeV GCT jet seed threshold
                      'L1GctJetFinderParams_GCTPhysics_2012_04_27_JetSeedThresh5GeV_mc,L1GctJetFinderParamsRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                      'L1HfRingEtScale_GCTPhysics_2012_04_27_JetSeedThresh5GeV_mc,L1HfRingEtScaleRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                      'L1HtMissScale_GCTPhysics_2012_04_27_JetSeedThresh5GeV_mc,L1HtMissScaleRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                      'L1JetEtScale_GCTPhysics_2012_04_27_JetSeedThresh5GeV_mc,L1JetEtScaleRcd,frontier://FrontierProd/CMS_COND_31X_L1T',
                      # load the 2012 v8 HLT jet energy corrections
                      'JetCorrectorParametersCollection_AK5Calo_2012_V8_hlt_mc,JetCorrectionsRecord,frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS,AK5CaloHLT',
                      'JetCorrectorParametersCollection_AK5PF_2012_V8_hlt_mc,JetCorrectionsRecord,frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS,AK5PFHLT',
                      'JetCorrectorParametersCollection_AK5PFchs_2012_V8_hlt_mc,JetCorrectionsRecord,frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS,AK5PFchsHLT' ),

    # global tag for MC production with realistic conditions 
    'startup'     :   'START52_V11D::All',

    # This should always be the GR_R_Vxx GT
    'com10'       :   'GR_R_52_V7::All',

    # 'hltonline' should be the same as same as 'com10' until a compatible GR_H_Vxx tag is available, 
    # then it should point to the GR_H_Vxx tag and override the connection string and pfnPrefix for use offline
    'hltonline'   :   'GR_H_V29D::All,frontier://FrontierProd/CMS_COND_31X_GLOBALTAG,frontier://FrontierProd/',

    # same as 'hltonline', but force loading a more recent L1 menu, for running the L1 emulator and HLT over old data
    'hltonline11' : ( 'GR_H_V29D::All,frontier://FrontierProd/CMS_COND_31X_GLOBALTAG,frontier://FrontierProd/',
                      'L1GtTriggerMenu_L1Menu_Collisions2012_v2_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T' ),

    # global tag for Heavy Ions MC production with realistic conditions 
    'starthi'     :   'STARTHI52_V9::All'
}
