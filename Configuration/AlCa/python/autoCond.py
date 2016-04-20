autoCond = { 

    ### NEW KEYS ###
    # GlobalTag for MC production with perfectly aligned and calibrated detector for Run1
    'run1_design'       :   '80X_mcRun1_design_v4',
    # GlobalTag for MC production (pp collisions) with realistic alignment and calibrations for Run1
    'run1_mc'           :   '80X_mcRun1_realistic_v4',
    # GlobalTag for MC production (Heavy Ions collisions) with realistic alignment and calibrations for Run1
    'run1_mc_hi'        :   '80X_mcRun1_HeavyIon_v4',
    # GlobalTag for MC production (p-Pb collisions) with realistic alignment and calibrations for Run1
    'run1_mc_pa'        :   '80X_mcRun1_pA_v4',
    # GlobalTag for MC production with perfectly aligned and calibrated detector for Run2
    'run2_design'       :   '80X_mcRun2_design_v12',
    # GlobalTag for MC production with pessimistic alignment and calibrations for Run2
    'run2_mc_50ns'      :   '80X_mcRun2_startup_v12',
    #GlobalTag for MC production with optimistic alignment and calibrations for Run2
    'run2_mc'           :   '80X_mcRun2_asymptotic_v12',
    # GlobalTag for MC production (cosmics) with starup-like alignment and calibrations for Run2, Strip tracker in peak mode
    'run2_mc_cosmics'   :   '80X_mcRun2cosmics_startup_peak_v11',
    # GlobalTag for MC production (Heavy Ions collisions) with optimistic alignment and calibrations for Run2
    'run2_mc_hi'        :   '80X_mcRun2_HeavyIon_v8',
    # GlobalTag for Run1 data reprocessing
    'run1_data'         :   '80X_dataRun2_v13',
    # GlobalTag for Run2 data reprocessing
    'run2_data'         :   '80X_dataRun2_v13',
    # GlobalTag for Run2 data relvals: allows customization to run with fixed L1 menu
    'run2_data_relval'  :   '80X_dataRun2_relval_v9',
    # GlobalTag for Run1 HLT: it points to the online GT
    'run1_hlt'          :   '80X_dataRun2_HLT_frozen_v11',
    # GlobalTag for Run2 HLT: it points to the online GT
    'run2_hlt'          :   '80X_dataRun2_HLT_frozen_v11',
    # GlobalTag for Run2 HLT RelVals: customizations to run with fixed L1 Menu
    'run2_hlt_relval'   :   '80X_dataRun2_HLT_relval_v8',
    # GlobalTag for Run2 HLT for HI: it points to the online GT
    'run2_hlt_hi'       :   '80X_dataRun2_HLTHI_frozen_v7',
    # GlobalTag for MC production with perfectly aligned and calibrated detector for Phase1 2017
    'phase1_2017_design' :  '80X_upgrade2017_design_v11',
    # GlobalTag for MC production with realistic conditions for for Phase1 2017 detector
    'phase1_2017_realistic': '80X_upgrade2017_realistic_v3',
    # GlobalTag for MC production with perfectly aligned and calibrated detector for Phase1 2019
    'phase1_2019_design'   : 'DES19_70_V2', # placeholder (GT not meant for standard RelVal) 
    # GlobalTag for MC production with perfectly aligned and calibrated detector for Phase2
    'phase2_design'        : 'POSTLS262_V1', # placeholder (GT not meant for standard RelVal)
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
