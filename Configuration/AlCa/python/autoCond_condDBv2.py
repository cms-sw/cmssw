autoCond = { 

    ### NEW KEYS ###
    # GlobalTag for MC production with perfectly aligned and calibrated detector for Run1
    'run1_design'       :   'PRE_MC_72_V5',
    # GlobalTag for MC production (pp collisions) with realistic alignment and calibrations for Run1
    'run1_mc'           :   'PRE_STA72_V5',
    # GlobalTag for MC production (Heavy Ions collisions) with realistic alignment and calibrations for Run1
    'run1_mc_hi'        :   'PRE_SHI72_V9',
    # GlobalTag for MC production (p-Pb collisions) with realistic alignment and calibrations for Run1
    'run1_mc_pa'        :   'PRE_SHI72_V10',
    # GlobalTag for MC production with perfectly aligned and calibrated detector for Run2
    'run2_design'       :   'PRE_DES72_V7',
    # GlobalTag for MC production with pessimistic alignment and calibrations for Run2
    'run2_mc_50ns'      :   'PRE_LS172_V14',
    #GlobalTag for MC production with optimistic alignment and calibrations for Run2
    'run2_mc'           :   'PRE_LS172_V13',
    # GlobalTag for Run1 data reprocessing
    'run1_data'         :   'PRE_R_72_V8A',
    # GlobalTag for Run2 data reprocessing
    'run2_data'         :   'PRE_R_72_V9A',
    # GlobalTag for Run1 HLT: it points to the online GT and overrides the connection string and pfnPrefix for use offline
    'run1_hlt'          :   'GR_H_V38A,frontier://FrontierProd/CMS_CONDITIONS,frontier://FrontierProd/',
    # GlobalTag for Run2 HLT: it points to the online GT and overrides the connection string and pfnPrefix for use offline
    'run2_hlt'          :   'GR_H_V40A,frontier://FrontierProd/CMS_CONDITIONS,frontier://FrontierProd/',
    # GlobalTag for MC production with perfectly aligned and calibrated detector for Phase1 2017
    'phase1_2017_design' :  'DES17_70_V2', # placeholder (GT not meant for standard RelVal)
    # GlobalTag for MC production with perfectly aligned and calibrated detector for Phase1 2019
    'phase1_2019_design' :  'DES19_70_V2', # placeholder (GT not meant for standard RelVal) 
    # GlobalTag for MC production with perfectly aligned and calibrated detector for Phase2
    'phase2_design'     :   'POSTLS262_V1', # placeholder (GT not meant for standard RelVal)
}

aliases = {
    'MAINGT' : 'FT_P_V42D|AN_V4',
    'BASEGT' : 'BASE1_V1|BASE2_V1'
}

# L1 configuration used during Run2012D
conditions_L1_Run2012D = (
    # L1 GT menu 2012 v3, used during Run2012D
    'L1GtTriggerMenu_L1Menu_Collisions2012_v3_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_CONDITIONS',
    # L1 GCT configuration with 5 GeV jet seed threshold, used since Run2012C
    'L1GctJetFinderParams_GCTPhysics_2012_04_27_JetSeedThresh5GeV_mc,L1GctJetFinderParamsRcd,frontier://FrontierProd/CMS_CONDITIONS',
    'L1HfRingEtScale_GCTPhysics_2012_04_27_JetSeedThresh5GeV_mc,L1HfRingEtScaleRcd,frontier://FrontierProd/CMS_CONDITIONS',
    'L1HtMissScale_GCTPhysics_2012_04_27_JetSeedThresh5GeV_mc,L1HtMissScaleRcd,frontier://FrontierProd/CMS_CONDITIONS',
    'L1JetEtScale_GCTPhysics_2012_04_27_JetSeedThresh5GeV_mc,L1JetEtScaleRcd,frontier://FrontierProd/CMS_CONDITIONS',
    # L1 CSCTF configuration used since Run2012B
    'L1MuCSCPtLut_key-11_mc,L1MuCSCPtLutRcd,frontier://FrontierProd/CMS_CONDITIONS',
    # L1 DTTF settings used since Run2012C
    'L1MuDTTFParameters_dttf12_TSC_03_csc_col_mc,L1MuDTTFParametersRcd,frontier://FrontierProd/CMS_CONDITIONS',
)

autoCond['run1_mc']          = ( autoCond['run1_mc'], ) \
                             + conditions_L1_Run2012D 
 
autoCond['run1_mc_hi']       = ( autoCond['run1_mc_hi'], ) \
                             + conditions_L1_Run2012D 
 
autoCond['run1_mc_pa']       = ( autoCond['run1_mc_pa'], ) \
                             + conditions_L1_Run2012D 
 
autoCond['run1_hlt']         = ( autoCond['run1_hlt'], ) \
                             + conditions_L1_Run2012D 
 
autoCond['run1_data']        = ( autoCond['run1_data'], ) \
                             + conditions_L1_Run2012D 
 
autoCond['run2_mc']          = ( autoCond['run2_mc'], ) \
                             + conditions_L1_Run2012D 
 
autoCond['run2_mc_50ns']     = ( autoCond['run2_mc_50ns'], ) \
                             + conditions_L1_Run2012D 

# dedicated GlobalTags for MC production with the fixed HLT menus
autoCond['run1_mc_2014']     = ( autoCond['run1_mc'] )
autoCond['run1_mc_Fake']     = ( autoCond['run1_mc'] )

autoCond['run1_mc_FULL']     = ( autoCond['run1_mc'] )
autoCond['run1_mc_GRun']     = ( autoCond['run1_mc'] )
autoCond['run1_mc_HIon']     = ( autoCond['run1_mc_hi'] )
autoCond['run1_mc_PIon']     = ( autoCond['run1_mc_pa'] )

autoCond['run2_mc_FULL']     = ( autoCond['run2_mc'] )
autoCond['run2_mc_GRun']     = ( autoCond['run2_mc'] )
autoCond['run2_mc_HIon']     = ( autoCond['run2_mc'] )
autoCond['run2_mc_PIon']     = ( autoCond['run2_mc'] )

# dedicated GlobalTags for running the fixed HLT menus on data
autoCond['run1_hlt_2014']    = ( autoCond['run1_hlt'] )
autoCond['run1_hlt_Fake']    = ( autoCond['run1_hlt'] )

autoCond['run1_hlt_FULL']    = ( autoCond['run1_hlt'] )
autoCond['run1_hlt_GRun']    = ( autoCond['run1_hlt'] )
autoCond['run1_hlt_HIon']    = ( autoCond['run1_hlt'] )
autoCond['run1_hlt_PIon']    = ( autoCond['run1_hlt'] )

autoCond['run2_hlt_FULL']    = ( autoCond['run2_hlt'] )
autoCond['run2_hlt_GRun']    = ( autoCond['run2_hlt'] )
autoCond['run2_hlt_HIon']    = ( autoCond['run2_hlt'] )
autoCond['run2_hlt_PIon']    = ( autoCond['run2_hlt'] )

# dedicated GlobalTags for running RECO and the fixed HLT menus on data
autoCond['run1_data_2014']    = ( autoCond['run1_data'] )
autoCond['run1_data_Fake']    = ( autoCond['run1_data'] )

autoCond['run1_data_FULL']    = ( autoCond['run1_data'] )
autoCond['run1_data_GRun']    = ( autoCond['run1_data'] )
autoCond['run1_data_HIon']    = ( autoCond['run1_data'] )
autoCond['run1_data_PIon']    = ( autoCond['run1_data'] )

autoCond['run2_data_FULL']    = ( autoCond['run2_data'] )
autoCond['run2_data_GRun']    = ( autoCond['run2_data'] )
autoCond['run2_data_HIon']    = ( autoCond['run2_data'] )
autoCond['run2_data_PIon']    = ( autoCond['run2_data'] )


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

### OLD KEYS ### for HLT                                                                                                                                          

autoCond['startup_2014']     = ( autoCond['run1_mc_2014'] )
autoCond['startup_GRun']     = ( autoCond['run1_mc_GRun'] )
autoCond['starthi_HIon']     = ( autoCond['run1_mc_HIon'] )
autoCond['startup_PIon']     = ( autoCond['run1_mc_PIon'] )

autoCond['hltonline_2014']   = ( autoCond['run1_hlt_2014'] )
autoCond['hltonline_GRun']   = ( autoCond['run1_hlt_GRun'] )
autoCond['hltonline_HIon']   = ( autoCond['run1_hlt_HIon'] )
autoCond['hltonline_PIon']   = ( autoCond['run1_hlt_PIon'] )

autoCond['com10_2014']       = ( autoCond['run1_data_2014'] )
autoCond['com10_GRun']       = ( autoCond['run1_data_GRun'] )
autoCond['com10_HIon']       = ( autoCond['run1_data_HIon'] )
autoCond['com10_PIon']       = ( autoCond['run1_data_PIon'] )
