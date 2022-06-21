autoCond = {

    ### NEW KEYS ###
    # GlobalTag for MC production with perfectly aligned and calibrated detector for Run1
    'run1_design'                  : '123X_mcRun1_design_v1',
    # GlobalTag for MC production (pp collisions) with realistic alignment and calibrations for Run1
    'run1_mc'                      : '123X_mcRun1_realistic_v1',
    # GlobalTag for MC production (Heavy Ions collisions) with realistic alignment and calibrations for Run1
    'run1_mc_hi'                   : '123X_mcRun1_HeavyIon_v1',
    # GlobalTag for MC production with pessimistic alignment and calibrations for Run2
    'run2_mc_50ns'                 : '123X_mcRun2_startup_v1',
    # GlobalTag for MC production (2015 L1 Trigger Stage1) with startup-like alignment and calibrations for Run2, L1 trigger in Stage1 mode
    'run2_mc_l1stage1'             : '123X_mcRun2_asymptotic_l1stage1_v1',
    # GlobalTag for MC production with perfectly aligned and calibrated detector for Run2
    'run2_design'                  : '123X_mcRun2_design_v1',
    #GlobalTag for MC production with optimistic alignment and calibrations for 2016, prior to VFP change
    'run2_mc_pre_vfp'              : '123X_mcRun2_asymptotic_preVFP_v1',
    #GlobalTag for MC production with optimistic alignment and calibrations for 2016, after VFP change
    'run2_mc'                      : '123X_mcRun2_asymptotic_v1',
    # GlobalTag for MC production (cosmics) with starup-like alignment and calibrations for Run2, Strip tracker in peak mode
    'run2_mc_cosmics'              : '123X_mcRun2cosmics_asymptotic_deco_v1',
    # GlobalTag for MC production (Heavy Ions collisions) with optimistic alignment and calibrations for Run2
    'run2_mc_hi'                   : '123X_mcRun2_HeavyIon_v1',
    # GlobalTag for MC production (p-Pb collisions) with realistic alignment and calibrations for Run2
    'run2_mc_pa'                   : '123X_mcRun2_pA_v1',
    # GlobalTag for Run2 data reprocessing
    'run2_data'                    : '124X_dataRun2_v2',
    # GlobalTag for Run2 data 2018B relvals only: HEM-15-16 fail
    'run2_data_HEfail'             : '124X_dataRun2_HEfail_v2',
    # GlobalTag for Run2 data relvals: allows customization to run with fixed L1 menu
    'run2_data_relval'             : '124X_dataRun2_relval_v2',
    # GlobalTag for Run2 HI data
    'run2_data_promptlike_hi'      : '124X_dataRun2_PromptLike_HI_v1',
    # GlobalTag with fixed snapshot time for Run2 HLT RelVals: customizations to run with fixed L1 Menu
    'run2_hlt_relval'              : '123X_dataRun2_HLT_relval_v3',
    # GlobalTag for Run3 HLT: identical to the online GT (124X_dataRun3_HLT_v1) but with snapshot at 2022-06-20 11:11:45 (UTC)
    'run3_hlt'                     : '124X_dataRun3_HLT_frozen_v2',
    # GlobalTag with fixed snapshot time for Run3 HLT RelVals: customizations to run with fixed L1 Menu
    'run3_hlt_relval'              : '124X_dataRun3_HLT_relval_v3',
    # GlobalTag for Run3 data relvals (express GT) - identical to 124X_dataRun3_Express_v1 but with snapshot at 2022-06-09 20:00:00 (UTC)
    'run3_data_express'            : '124X_dataRun3_Express_frozen_v1',
    # GlobalTag for Run3 data relvals (prompt GT) - identical to 124X_dataRun3_Prompt_v1 but with snapshot at 2022-06-09 20:00:00 (UTC)
    'run3_data_prompt'             : '124X_dataRun3_Prompt_frozen_v1',
    # GlobalTag for Run3 offline data reprocessing - snapshot updated to 2022-06-20 11:11:45 (UTC)
    'run3_data'                    : '124X_dataRun3_v4',
    # GlobalTag for Run3 data relvals: allows customization to run with fixed L1 menu
    'run3_data_relval'             : '124X_dataRun3_relval_v4',
    # GlobalTag for MC production with perfectly aligned and calibrated detector for Phase1 2017 (and 0,0,~0-centred beamspot)
    'phase1_2017_design'           : '123X_mc2017_design_v2',
    # GlobalTag for MC production with realistic conditions for Phase1 2017 detector
    'phase1_2017_realistic'        : '123X_mc2017_realistic_v2',
    # GlobalTag for MC production (cosmics) with realistic alignment and calibrations for Phase1 2017 detector, Strip tracker in DECO mode
    'phase1_2017_cosmics'          : '123X_mc2017cosmics_realistic_deco_v2',
    # GlobalTag for MC production (cosmics) with realistic alignment and calibrations for Phase1 2017 detector, Strip tracker in PEAK mode
    'phase1_2017_cosmics_peak'     : '123X_mc2017cosmics_realistic_peak_v2',
    # GlobalTag for MC production with perfectly aligned and calibrated detector for full Phase1 2018 (and 0,0,0-centred beamspot)
    'phase1_2018_design'           : '123X_upgrade2018_design_v4',
    # GlobalTag for MC production with realistic conditions for full Phase1 2018 detector
    'phase1_2018_realistic'        : '123X_upgrade2018_realistic_v2',
    # GlobalTag for MC production with realistic run-dependent (RD) conditions for full Phase1 2018 detector
    'phase1_2018_realistic_rd'     : '123X_upgrade2018_realistic_RD_v3',
    # GlobalTag for MC production with realistic conditions for full Phase1 2018 detector for Heavy Ion
    'phase1_2018_realistic_hi'     : '123X_upgrade2018_realistic_HI_v2',
    # GlobalTag for MC production with realistic conditions for full Phase1 2018 detector: HEM-15-16 fail
    'phase1_2018_realistic_HEfail' : '123X_upgrade2018_realistic_HEfail_v2',
    # GlobalTag for MC production (cosmics) with realistic conditions for full Phase1 2018 detector,  Strip tracker in DECO mode
    'phase1_2018_cosmics'          : '123X_upgrade2018cosmics_realistic_deco_v2',
    # GlobalTag for MC production (cosmics) with realistic conditions for full Phase1 2018 detector,  Strip tracker in PEAK mode
    'phase1_2018_cosmics_peak'     : '123X_upgrade2018cosmics_realistic_peak_v2',
    # GlobalTag for MC production with perfectly aligned and calibrated detector for Phase1 2022
    'phase1_2022_design'           : '124X_mcRun3_2022_design_v6',
    # GlobalTag for MC production with realistic conditions for Phase1 2022
    'phase1_2022_realistic'        : '124X_mcRun3_2022_realistic_v6',
    # GlobalTag for MC production (cosmics) with realistic conditions for Phase1 2022,  Strip tracker in DECO mode
    'phase1_2022_cosmics'          : '124X_mcRun3_2022cosmics_realistic_deco_v7',
    # GlobalTag for MC production (cosmics) with perfectly aligned and calibrated detector for Phase1 2022, Strip tracker in DECO mode
    'phase1_2022_cosmics_design'   : '124X_mcRun3_2022cosmics_design_deco_v6',
    # GlobalTag for MC production with realistic conditions for Phase1 2022 detector for Heavy Ion
    'phase1_2022_realistic_hi'     : '124X_mcRun3_2022_realistic_HI_v6',
    # GlobalTag for MC production with realistic conditions for Phase1 2023
    'phase1_2023_realistic'        : '124X_mcRun3_2023_realistic_v6',
    # GlobalTag for MC production with realistic conditions for Phase1 2024
    'phase1_2024_realistic'        : '124X_mcRun3_2024_realistic_v6',
    # GlobalTag for MC production with realistic conditions for Phase2
    'phase2_realistic'             : '124X_mcRun4_realistic_v7'
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

# special GT for Run3 DDD geometry
from Configuration.AlCa.autoCondModifiers import autoCondDDD
autoCond = autoCondDDD(autoCond)

# special GT for 2017 ppRef 5 TeV
from Configuration.AlCa.autoCondModifiers import autoCond2017ppRef5TeV
autoCond = autoCond2017ppRef5TeV(autoCond)


### OLD KEYS ### kept for backward compatibility
    # GlobalTag for MC production with perfectly aligned and calibrated detector
autoCond['mc']               = ( autoCond['run1_design'] )
    # GlobalTag for MC production with realistic alignment and calibrations
autoCond['startup']          = ( autoCond['run1_mc'] )
    # GlobalTag for MC production of Heavy Ions events with realistic alignment and calibrations
autoCond['starthi']          = ( autoCond['run1_mc_hi'] )
    # GlobalTag for data reprocessing
autoCond['com10']            = ( autoCond['run1_data'] )
    # GlobalTag for running HLT on recent data: it points to the online GT (remove the snapshot!)
autoCond['hltonline']        = ( autoCond['run3_hlt'] )
    # GlobalTag for POSTLS1 upgrade studies:
autoCond['upgradePLS1']      = ( autoCond['run2_mc'] )
autoCond['upgradePLS150ns']  = ( autoCond['run2_mc_50ns'] )
autoCond['upgrade2017']      = ( autoCond['phase1_2017_design'] )
autoCond['upgrade2021']      = ( autoCond['phase1_2022_design'] )
autoCond['upgrade2022']      = ( autoCond['phase1_2022_design'] )
autoCond['upgradePLS3']      = ( autoCond['phase2_realistic'] )
