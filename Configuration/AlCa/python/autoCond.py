autoCond = { 

    ### NEW KEYS ###
    # GlobalTag for MC production with perfectly aligned and calibrated detector for Run1
    'run1_design'       :   'MC_72_V1::All',
    # GlobalTag for MC production (pp collisions) with realistic alignment and calibrations for Run1
    'run1_mc'           :   'START72_V1::All',
    # GlobalTag for MC production (Heavy Ions collisions) with realistic alignment and calibrations for Run1
    'run1_mc_hi'        :   'STARTHI72_V1::All',
    # GlobalTag for MC production (p-Pb collisions) with realistic alignment and calibrations for Run1
    'run1_mc_pa'        :   'STARTHI72_V2::All',
    # GlobalTag for MC production with perfectly aligned and calibrated detector for Run2
    'run2_design'       :   'DESIGN72_V2::All',
    # GlobalTag for MC production with pessimistic alignment and calibrations for Run2
    'run2_mc_50ns'      :   'POSTLS172_V4::All',
    #GlobalTag for MC production with optimistic alignment and calibrations for Run2
    'run2_mc'           :   'POSTLS172_V3::All',
    # GlobalTag for Run1 data reprocessing
    'run1_data'         :   'GR_R_72_V2::All',
    # GlobalTag for Run2 data reprocessing
    'run2_data'         :   'GR_R_72_V2::All',
    # GlobalTag for Run1 HLT: it points to the online GT and overrides the connection string and pfnPrefix for use offline
    'run1_hlt'          :   'GR_H_V37::All,frontier://FrontierProd/CMS_COND_31X_GLOBALTAG,frontier://FrontierProd/',
    # GlobalTag for Run2 HLT: it points to the online GT and overrides the connection string and pfnPrefix for use offline
    'run2_hlt'          :   'GR_H_V39::All,frontier://FrontierProd/CMS_COND_31X_GLOBALTAG,frontier://FrontierProd/',
    # GlobalTag for MC production with perfectly aligned and calibrated detector for Phase1 2017
    'phase1_2017_design' :  'DES17_70_V2::All', # placeholder (GT not meant for standard RelVal)
    # GlobalTag for MC production with perfectly aligned and calibrated detector for Phase1 2019
    'phase1_2019_design' :  'DES19_70_V2::All', # placeholder (GT not meant for standard RelVal) 
    # GlobalTag for MC production with perfectly aligned and calibrated detector for Phase2
    'phase2_design'     :   'POSTLS262_V1::All', # placeholder (GT not meant for standard RelVal)
}

aliases = {
    'MAINGT' : 'FT_P_V42D::All|AN_V4::All',
    'BASEGT' : 'BASE1_V1::All|BASE2_V1::All'
}

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

# HLT Jet Energy Corrections
conditions_HLT_JECs = (
    # HLT 2012 jet energy corrections
    'JetCorrectorParametersCollection_Jec11_V12_AK5CaloHLT,JetCorrectionsRecord,frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS,AK5CaloHLT',
    'JetCorrectorParametersCollection_AK5PF_2012_V8_hlt_mc,JetCorrectionsRecord,frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS,AK5PFHLT',
    'JetCorrectorParametersCollection_AK5PFchs_2012_V8_hlt_mc,JetCorrectionsRecord,frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS,AK5PFchsHLT',
    # HLT 2014 jet energy corrections - tk0 scenario
    'JetCorrectorParametersCollection_HLT_V1_AK4Calo,JetCorrectionsRecord,frontier://FrontierPrep/CMS_COND_PHYSICSTOOLS,AK4CaloHLT',
    'JetCorrectorParametersCollection_HLT_trk0_V1_AK4PF,JetCorrectionsRecord,frontier://FrontierPrep/CMS_COND_PHYSICSTOOLS,AK4PFHLT',
)

autoCond['run1_mc']          = ( autoCond['run1_mc'], ) \
                             + conditions_L1_Run2012D \
                             + conditions_HLT_JECs
 
autoCond['run1_mc_hi']       = ( autoCond['run1_mc_hi'], ) \
                             + conditions_L1_Run2012D \
                             + conditions_HLT_JECs
 
autoCond['run1_mc_pa']       = ( autoCond['run1_mc_pa'], ) \
                             + conditions_L1_Run2012D \
                             + conditions_HLT_JECs
 
autoCond['run1_hlt']         = ( autoCond['run1_hlt'], ) \
                             + conditions_L1_Run2012D \
                             + conditions_HLT_JECs
 
autoCond['run1_data']        = ( autoCond['run1_data'], ) \
                             + conditions_L1_Run2012D \
                             + conditions_HLT_JECs
 
autoCond['run2_mc']          = ( autoCond['run2_mc'], ) \
                             + conditions_L1_Run2012D \
                             + conditions_HLT_JECs
 
autoCond['run2_mc_50ns']     = ( autoCond['run2_mc_50ns'], ) \
                             + conditions_L1_Run2012D \
                             + conditions_HLT_JECs
 
# dedicated GlobalTags for MC production with the fixed HLT menus
autoCond['startup_2014']     = ( autoCond['run1_mc'] )

autoCond['startup_GRun']     = ( autoCond['run1_mc'] )

autoCond['starthi_HIon']     = ( autoCond['run1_mc_hi'] )

autoCond['startup_PIon']     = ( autoCond['run1_mc_pa'] )

# dedicated GlobalTags for running the fixed HLT menus on data
autoCond['hltonline_2014']   = ( autoCond['run1_hlt'] )

autoCond['hltonline_GRun']   = ( autoCond['run1_hlt'] )

autoCond['hltonline_HIon']   = ( autoCond['run1_hlt'] )

autoCond['hltonline_PIon']   = ( autoCond['run1_hlt'] )

# dedicated GlobalTags for running RECO and the fixed HLT menus on data
autoCond['com10_2014']       = ( autoCond['run1_data'] )

autoCond['com10_GRun']       = ( autoCond['run1_data'] )

autoCond['com10_HIon']       = ( autoCond['run1_data'] )

autoCond['com10_PIon']       = ( autoCond['run1_data'] )


### OLD KEYS ### kept for backward compatibility
    # GlobalTag for MC production with perfectly aligned and calibrated detector
autoCond['mc']               = ( autoCond['run1_design'] )
    # GlobalTag for MC production with realistic alignment and calibrations
autoCond['startup']          = ( autoCond['run1_mc'] )
    # GlobalTag for MC production of Heavy Ions events with realistic alignment and calibrations
autoCond['starthi']          = ( autoCond['run1_mc_hi'] )
    # GlobalTag for MC production of p-Pb events with realistic alignment and calibrations
autoCond['startpa']          = ( autoCond['run1_mc_pa'] )
    # GlobalTag for data reprocessing: this should always be the GR_R tag
autoCond['com10']            = ( autoCond['run1_data'] )
    # GlobalTag for running HLT on recent data: this should be the GR_P (prompt reco) global tag until a compatible GR_H tag is available, 
    # then it should point to the GR_H tag and override the connection string and pfnPrefix for use offline
autoCond['hltonline']        = ( autoCond['run1_hlt'] )
    # GlobalTag for POSTLS1 upgrade studies:
autoCond['upgradePLS1']      = ( autoCond['run2_mc'] )
autoCond['upgradePLS150ns']  = ( autoCond['run2_mc_50ns'] )
autoCond['upgrade2017']      = ( autoCond['phase1_2017_design'] )
autoCond['upgrade2019']      = ( autoCond['phase1_2019_design'] )
autoCond['upgradePLS3']      = ( autoCond['phase2_design'] )
