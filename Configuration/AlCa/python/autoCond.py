autoCond = { 

    ### NEW KEYS ###
    # GlobalTag for MC production with perfectly aligned and calibrated detector for Run1
    'run1_design'       :   '93X_mcRun1_design_v0',
    # GlobalTag for MC production (pp collisions) with realistic alignment and calibrations for Run1
    'run1_mc'           :   '93X_mcRun1_realistic_v0',
    # GlobalTag for MC production (Heavy Ions collisions) with realistic alignment and calibrations for Run1
    'run1_mc_hi'        :   '93X_mcRun1_HeavyIon_v0',
    # GlobalTag for MC production (p-Pb collisions) with realistic alignment and calibrations for Run1
    'run1_mc_pa'        :   '93X_mcRun1_pA_v0',
    # GlobalTag for MC production with perfectly aligned and calibrated detector for Run2
    'run2_design'       :   '93X_mcRun2_design_v0',
    # GlobalTag for MC production with pessimistic alignment and calibrations for Run2
    'run2_mc_50ns'      :   '93X_mcRun2_startup_v0',
    #GlobalTag for MC production with optimistic alignment and calibrations for Run2
    'run2_mc'           :   '93X_mcRun2_asymptotic_v0',
    # GlobalTag for MC production (cosmics) with starup-like alignment and calibrations for Run2, Strip tracker in peak mode
    'run2_mc_cosmics'   :   '93X_mcRun2cosmics_startup_deco_v0',
    # GlobalTag for MC production (Heavy Ions collisions) with optimistic alignment and calibrations for Run2
    'run2_mc_hi'        :   '93X_mcRun2_HeavyIon_v0',
    # GlobalTag for MC production (p-Pb collisions) with realistic alignment and calibrations for Run2
    'run2_mc_pa'        :   '93X_mcRun2_pA_v0',
    # GlobalTag for Run1 data reprocessing
    'run1_data'         :   '93X_dataRun2_v0',
    # GlobalTag for Run2 data reprocessing
    'run2_data'         :   '93X_dataRun2_v0',
    # GlobalTag for Run2 data relvals: allows customization to run with fixed L1 menu
    'run2_data_relval'  :   '93X_dataRun2_relval_v0',
    # GlobalTag for Run2 data 2016H relvals only: Prompt Conditions + fixed L1 menu (to be removed)
    'run2_data_promptlike' : '93X_dataRun2_PromptLike_v0',
    # GlobalTag for Run1 HLT: it points to the online GT
    'run1_hlt'          :   '93X_dataRun2_HLT_frozen_v0',
    # GlobalTag for Run2 HLT: it points to the online GT
    'run2_hlt'          :   '93X_dataRun2_HLT_frozen_v0',
    # GlobalTag for Run2 HLT RelVals: customizations to run with fixed L1 Menu
    'run2_hlt_relval'   :   '93X_dataRun2_HLT_relval_v1',
    # GlobalTag for Run2 HLT for HI: it points to the online GT
    'run2_hlt_hi'       :   '93X_dataRun2_HLTHI_frozen_v0',
    # GlobalTag for MC production with perfectly aligned and calibrated detector for Phase1 2017 (and 0,0,~0-centred beamspot)
    'phase1_2017_design'       :  '93X_mc2017_design_IdealBS_v2',
    # GlobalTag for MC production with realistic conditions for Phase1 2017 detector
    'phase1_2017_realistic'    : '93X_mc2017_realistic_v2',
    # GlobalTag for MC production (cosmics) with realistic alignment and calibrations for Phase1 2017 detector, Strip tracker in DECO mode
    'phase1_2017_cosmics'      : '93X_mc2017cosmics_realistic_deco_v2',
    # GlobalTag for MC production (cosmics) with realistic alignment and calibrations for Phase1 2017 detector, Strip tracker in PEAK mode
    'phase1_2017_cosmics_peak' : '93X_mc2017cosmics_realistic_peak_v2',
    # GlobalTag for MC production with perfectly aligned and calibrated detector for full Phase1 2018 (and 0,0,0-centred beamspot)
    'phase1_2018_design'       : '93X_upgrade2018_design_IdealBS_v1',
    # GlobalTag for MC production with realistic conditions for full Phase1 2018 detector
    'phase1_2018_realistic'    : '93X_upgrade2018_realistic_v1',
    # GlobalTag for MC production (cosmics) with realistic conditions for full Phase1 2018 detector,  Strip tracker in DECO mode
    'phase1_2018_cosmics'      :   '93X_upgrade2018cosmics_realistic_deco_v1',
    # GlobalTag for MC production with perfectly aligned and calibrated detector for Phase1 2019
    'phase1_2019_design'       : 'DES19_70_V2', # placeholder (GT not meant for standard RelVal) 
    # GlobalTag for MC production with realistic conditions for Phase2 2023
    'phase2_realistic'         : '93X_upgrade2023_realistic_v1'
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
    # GlobalTag for data reprocessing
autoCond['com10']            = ( autoCond['run1_data'] )
    # GlobalTag for running HLT on recent data: it points to the online GT (remove the snapshot!)
autoCond['hltonline']        = ( autoCond['run1_hlt'] )
    # GlobalTag for POSTLS1 upgrade studies:
autoCond['upgradePLS1']      = ( autoCond['run2_mc'] )
autoCond['upgradePLS150ns']  = ( autoCond['run2_mc_50ns'] )
autoCond['upgrade2017']      = ( autoCond['phase1_2017_design'] )
autoCond['upgrade2019']      = ( autoCond['phase1_2019_design'] )
autoCond['upgradePLS3']      = ( autoCond['phase2_realistic'] )
