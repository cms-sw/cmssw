autoCond = {

    ### NEW KEYS ###
    # GlobalTag for MC production with perfectly aligned and calibrated detector for Run1
    'run1_design'                  : '113X_mcRun1_design_v3',
    # GlobalTag for MC production (pp collisions) with realistic alignment and calibrations for Run1
    'run1_mc'                      : '113X_mcRun1_realistic_v3',
    # GlobalTag for MC production (Heavy Ions collisions) with realistic alignment and calibrations for Run1
    'run1_mc_hi'                   : '113X_mcRun1_HeavyIon_v3',
    # GlobalTag for MC production (p-Pb collisions) with realistic alignment and calibrations for Run1
    'run1_mc_pa'                   : '113X_mcRun1_pA_v3',
    # GlobalTag for MC production with pessimistic alignment and calibrations for Run2
    'run2_mc_50ns'                 : '113X_mcRun2_startup_v3',
    # GlobalTag for MC production (2015 L1 Trigger Stage1) with startup-like alignment and calibrations for Run2, L1 trigger in Stage1 mode
    'run2_mc_l1stage1'             : '113X_mcRun2_asymptotic_l1stage1_v4',
    # GlobalTag for MC production with perfectly aligned and calibrated detector for Run2
    'run2_design'                  : '113X_mcRun2_design_v4',
    #GlobalTag for MC production with optimistic alignment and calibrations for 2016, prior to VFP change
    'run2_mc_pre_vfp'              : '120X_mcRun2_asymptotic_preVFP_v1',
    #GlobalTag for MC production with optimistic alignment and calibrations for 2016, after VFP change
    'run2_mc'                      : '120X_mcRun2_asymptotic_v1',
    # GlobalTag for MC production (cosmics) with starup-like alignment and calibrations for Run2, Strip tracker in peak mode
    'run2_mc_cosmics'              : '113X_mcRun2cosmics_asymptotic_deco_v4',
    # GlobalTag for MC production (Heavy Ions collisions) with optimistic alignment and calibrations for Run2
    'run2_mc_hi'                   : '113X_mcRun2_HeavyIon_v4',
    # GlobalTag for MC production (p-Pb collisions) with realistic alignment and calibrations for Run2
    'run2_mc_pa'                   : '113X_mcRun2_pA_v4',
    # GlobalTag for Run2 data reprocessing
    'run2_data'                    : '120X_dataRun2_v2',
    # GlobalTag for Run2 data 2018B relvals only: HEM-15-16 fail
    'run2_data_HEfail'             : '120X_dataRun2_HEfail_v1',
    # GlobalTag for Run2 data relvals: allows customization to run with fixed L1 menu
    'run2_data_relval'             : '120X_dataRun2_relval_v2',
    # GlobalTag for Run2 HI data
    'run2_data_promptlike_hi'      : '120X_dataRun2_PromptLike_HI_v1',
    # GlobalTag for Run3 HLT: it points to the online GT
    'run3_hlt'                     : '113X_dataRun3_HLT_v3',
    # GlobalTag with fixed snapshot time for Run2 HLT RelVals: customizations to run with fixed L1 Menu
    'run2_hlt_relval'              : '113X_dataRun2_HLT_relval_v2',
    # GlobalTag for Run3 data relvals (express GT)
    'run3_data_express'            : '113X_dataRun3_Express_v4',
    # GlobalTag for Run3 data relvals
    'run3_data_prompt'             : '113X_dataRun3_Prompt_v3',
    # GlobalTag for MC production with perfectly aligned and calibrated detector for Phase1 2017 (and 0,0,~0-centred beamspot)
    'phase1_2017_design'           : '113X_mc2017_design_v5',
    # GlobalTag for MC production with realistic conditions for Phase1 2017 detector
    'phase1_2017_realistic'        : '113X_mc2017_realistic_v5',
    # GlobalTag for MC production with realistic conditions for Phase1 2017 detector, for PP reference run
    'phase1_2017_realistic_ppref'  :  '120X_mc2017_realistic_forppRef5TeV_v1',
    # GlobalTag for MC production (cosmics) with realistic alignment and calibrations for Phase1 2017 detector, Strip tracker in DECO mode
    'phase1_2017_cosmics'          : '113X_mc2017cosmics_realistic_deco_v5',
    # GlobalTag for MC production (cosmics) with realistic alignment and calibrations for Phase1 2017 detector, Strip tracker in PEAK mode
    'phase1_2017_cosmics_peak'     : '113X_mc2017cosmics_realistic_peak_v5',
    # GlobalTag for MC production with perfectly aligned and calibrated detector for full Phase1 2018 (and 0,0,0-centred beamspot)
    'phase1_2018_design'           : '113X_upgrade2018_design_v5',
    # GlobalTag for MC production with realistic conditions for full Phase1 2018 detector
    'phase1_2018_realistic'        : '113X_upgrade2018_realistic_v5',
    # GlobalTag for MC production with realistic run-dependent (RD) conditions for full Phase1 2018 detector
    'phase1_2018_realistic_rd'     : '113X_upgrade2018_realistic_RD_v4',
    # GlobalTag for MC production with realistic conditions for full Phase1 2018 detector for Heavy Ion
    'phase1_2018_realistic_hi'     : '113X_upgrade2018_realistic_HI_v5',
    # GlobalTag for MC production with realistic conditions for full Phase1 2018 detector: HEM-15-16 fail
    'phase1_2018_realistic_HEfail' : '113X_upgrade2018_realistic_HEfail_v5',
    # GlobalTag for MC production (cosmics) with realistic conditions for full Phase1 2018 detector,  Strip tracker in DECO mode
    'phase1_2018_cosmics'          : '113X_upgrade2018cosmics_realistic_deco_v5',
    # GlobalTag for MC production (cosmics) with realistic conditions for full Phase1 2018 detector,  Strip tracker in PEAK mode
    'phase1_2018_cosmics_peak'     : '113X_upgrade2018cosmics_realistic_peak_v5',
    # GlobalTag for MC production with perfectly aligned and calibrated detector for Phase1 2021
    'phase1_2021_design'           : '120X_mcRun3_2021_design_v2', # GT containing design conditions for Phase1 2021
    # GlobalTag for MC production with realistic conditions for Phase1 2021
    'phase1_2021_realistic'        : '120X_mcRun3_2021_realistic_v3', # GT containing realistic conditions for Phase1 2021
    # GlobalTag for MC production (cosmics) with realistic conditions for Phase1 2021,  Strip tracker in DECO mode
    'phase1_2021_cosmics'          : '120X_mcRun3_2021cosmics_realistic_deco_v2',
    # GlobalTag for MC production with realistic conditions for Phase1 2021 detector for Heavy Ion
    'phase1_2021_realistic_hi'     : '120X_mcRun3_2021_realistic_HI_v2',
    # GlobalTag for MC production with realistic conditions for Phase1 2023
    'phase1_2023_realistic'        : '120X_mcRun3_2023_realistic_v3', # GT containing realistic conditions for Phase1 2023
    # GlobalTag for MC production with realistic conditions for Phase1 2024
    'phase1_2024_realistic'        : '120X_mcRun3_2024_realistic_v3', # GT containing realistic conditions for Phase1 2024
    # GlobalTag for MC production with realistic conditions for Phase2
    'phase2_realistic'             : '113X_mcRun4_realistic_v7'
}

aliases = {
    'MAINGT' : 'FT_P_V42D|AN_V4',
    'BASEGT' : 'BASE1_V1|BASE2_V1'
}

### Run 1 data GTs ###
    # GlobalTag with fixed snapshot time for Run1 HLT RelVals: customizations to run with fixed L1 Menu
autoCond['run1_hlt_relval']  = autoCond['run2_hlt_relval']
    # GlobalTag for Run1 data reprocessing
autoCond['run1_data']        = autoCond['run2_data']

# dedicated GlobalTags for HLT
from Configuration.HLT.autoCondHLT import autoCondHLT
autoCond = autoCondHLT(autoCond)

# dedicated GlobalTags for phase-2 (specializing conditions for each geometry)
from Configuration.AlCa.autoCondPhase2 import autoCondPhase2
autoCond = autoCondPhase2(autoCond)

# special cases modifier for autoCond GTs
from Configuration.AlCa.autoCondModifiers import autoCond0T
autoCond = autoCond0T(autoCond)

# special GT for 2015 HLT HI run
from Configuration.AlCa.autoCondModifiers import autoCondHLTHI
autoCond = autoCondHLTHI(autoCond)

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
autoCond['hltonline']        = ( autoCond['run3_hlt'] )
    # GlobalTag for POSTLS1 upgrade studies:
autoCond['upgradePLS1']      = ( autoCond['run2_mc'] )
autoCond['upgradePLS150ns']  = ( autoCond['run2_mc_50ns'] )
autoCond['upgrade2017']      = ( autoCond['phase1_2017_design'] )
autoCond['upgrade2021']      = ( autoCond['phase1_2021_design'] )
autoCond['upgradePLS3']      = ( autoCond['phase2_realistic'] )
