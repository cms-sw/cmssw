autoCond = { 

    ### NEW KEYS ###
    # GlobalTag for MC production with perfectly aligned and calibrated detector for Run1
    'run1_design'       :   'PRE_MC_72_V4::All',
    # GlobalTag for MC production (pp collisions) with realistic alignment and calibrations for Run1
    'run1_mc'           :   'PRE_STA72_V4::All',
    # GlobalTag for MC production (Heavy Ions collisions) with realistic alignment and calibrations for Run1
    'run1_mc_hi'        :   'PRE_SHI72_V7::All',
    # GlobalTag for MC production (p-Pb collisions) with realistic alignment and calibrations for Run1
    'run1_mc_pa'        :   'PRE_SHI72_V8::All',
    # GlobalTag for MC production with perfectly aligned and calibrated detector for Run2
    'run2_design'       :   'PRE_DES72_V6::All',
    # GlobalTag for MC production with pessimistic alignment and calibrations for Run2
    'run2_mc_50ns'      :   'PRE_LS172_V12::All',
    #GlobalTag for MC production with optimistic alignment and calibrations for Run2
    'run2_mc'           :   'PRE_LS172_V11::All',
    # GlobalTag for Run1 data reprocessing
    'run1_data'         :   'PRE_R_72_V6A::All',
    # GlobalTag for Run2 data reprocessing
    'run2_data'         :   'PRE_R_72_V7A::All',
    # GlobalTag for Run1 HLT: it points to the online GT and overrides the connection string and pfnPrefix for use offline
    'run1_hlt'          :   'GR_H_V38A::All,frontier://FrontierProd/CMS_COND_31X_GLOBALTAG,frontier://FrontierProd/',
    # GlobalTag for Run2 HLT: it points to the online GT and overrides the connection string and pfnPrefix for use offline
    'run2_hlt'          :   'GR_H_V40A::All,frontier://FrontierProd/CMS_COND_31X_GLOBALTAG,frontier://FrontierProd/',
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
