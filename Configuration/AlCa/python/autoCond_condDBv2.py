autoCond = { 

    ### NEW KEYS ###
    # GlobalTag for MC production with perfectly aligned and calibrated detector for Run1
    'run1_design'       :   'DESRUN1_74_V6',
    # GlobalTag for MC production (pp collisions) with realistic alignment and calibrations for Run1
    'run1_mc'           :   'MCRUN1_74_V6',
    # GlobalTag for MC production (Heavy Ions collisions) with realistic alignment and calibrations for Run1
    'run1_mc_hi'        :   'MCHI1_74_V6',
    # GlobalTag for MC production (p-Pb collisions) with realistic alignment and calibrations for Run1
    'run1_mc_pa'        :   'MCPA1_74_V6',
    # GlobalTag for MC production with perfectly aligned and calibrated detector for Run2
    'run2_design'       :   'DESRUN2_74_V4',
    # GlobalTag for MC production with pessimistic alignment and calibrations for Run2
    'run2_mc_50ns'      :   'MCRUN2_74_V8',
    #GlobalTag for MC production with optimistic alignment and calibrations for Run2
    'run2_mc'           :   'MCRUN2_74_V9',
    # GlobalTag for MC production (Heavy Ions collisions) with optimistic alignment and calibrations for Run2
    'run2_mc_hi'        :   'MCHI2_74_V5',
    # GlobalTag for Run1 data reprocessing
    'run1_data'         :   'GR_R_74_V12A',
    # GlobalTag for Run2 data reprocessing
    'run2_data'         :   'GR_R_74_V13A',
    # GlobalTag for Run1 HLT: it points to the online GT and overrides the connection string and pfnPrefix for use offline
    'run1_hlt'          :   'GR_H_V57A,frontier://FrontierProd/CMS_CONDITIONS,frontier://FrontierProd/',
    # GlobalTag for Run2 HLT: it points to the online GT and overrides the connection string and pfnPrefix for use offline
    'run2_hlt'          :   'GR_H_V58A,frontier://FrontierProd/CMS_CONDITIONS,frontier://FrontierProd/',
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

# dedicated GlobalTags for HLT
from Configuration.HLT.autoCondHLT import autoCondHLT
autoCond = autoCondHLT(autoCond)

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
