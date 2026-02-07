from Configuration.PyReleaseValidation.relval_steps import *
import math


class WFN:
    def __init__(self, offset: int):
        self.offset = offset
        self.index = 1
        self.thousands = 0
        self.hundreds = 0

    def __call__(self) -> float:
        if self.index >= 100:
            raise ValueError("Overflow: Leading-index exceeded limit (100)")
        value_str = f"{self.offset}.{self.thousands}{self.hundreds}{self.index:02d}"
        result = float(value_str)
        self.index += 1
        return result

    def subnext(self) -> None:
        self.hundreds += 1
        self.index = 1
        if self.hundreds >= 10:
            raise ValueError("Overflow: Sub-index (hundreds) exceeded limit (999)")

    def next(self, index: int = None) -> None:
        if index is None:
            self.thousands += 1
            self.hundreds = 0
        else:
            if index <= self.thousands:
                raise ValueError("New index must be greater than current index")
            self.thousands = index
            self.hundreds = 0

        if self.thousands >= 10:
            raise ValueError("Overflow: Sub-sub-index (thousands) exceeded limit (9999)")
        self.index = 1


workflows = Matrix()

_NANO_data = merge([{'-s': 'NANO,DQM:@nanoAODDQM',
                     '--process': 'NANO',
                     '--data': '',
                     '--eventcontent': 'NANOAOD,DQM',
                     '--datatier': 'NANOAOD,DQMIO',
                     '-n': '10000',
                     '--customise': '"Configuration/DataProcessing/Utils.addMonitoring"'
                     }])
_HARVEST_nano = merge([{'-s': 'HARVESTING:@nanoAODDQM',
                        '--filetype': 'DQM',
                        '--filein': 'file:step2_inDQM.root',
                        '--conditions': 'auto:run2_data'  # this is fake for harvesting
                        }])
_HARVEST_data = merge([_HARVEST_nano, {'--data': ''}])


run2_lumis = {277168: [[1, 1708]],
              277194: [[913, 913], [916, 916], [919, 919], [932, 932], [939, 939]],
              283877: [[1, 1496]],
              299649: [[155, 332]],
              303885: [[60, 2052]],
              305040: [[200, 700]],
              305050: [[200, 700]],
              305234: [[1, 200]],
              305377: [[1, 500]],
              315489: [[1, 100]],
              320822: [[1, 200]],
              }
run3_lumis = {}

_NANO_mc = merge([{'-s': 'NANO,DQM:@nanoAODDQM',
                   '--process': 'NANO',
                   '--mc': '',
                   '--eventcontent': 'NANOAODSIM,DQM',
                   '--datatier': 'NANOAODSIM,DQMIO',
                   '-n': '10000',
                   '--customise': '"Configuration/DataProcessing/Utils.addMonitoring"'
                   }])
_HARVEST_mc = merge([_HARVEST_nano, {'--mc': ''}])
## the validation sequence VALIDATION:@miniAODValidation will not function on M2M, because it requires AOD-bound collections TBF
## the dqm sequence DQM:@miniAODDQM will not function on M2M, because it requires AOD-bound collections TBF
#_MINI_from_MINI = {'-s': 'PAT:Configuration/StandardSequences/REMINI_cff.patAlgosToolsTask,DQM:@miniAODDQM',
#                   '--process': 'M2M'}
_MINI_from_MINI = {'-s': 'PAT:Configuration/StandardSequences/REMINI_cff.patAlgosToolsTask',
                   '--eventcontent' : 'MINIAODSIM',
                   '--datatier' : 'MINIAODSIM',
                   '--process': 'M2M'}
_MINI_from_MINI_data = merge([{'--data': '',
                               '--eventcontent' : 'MINIAOD',
                               '--datatier' : 'MINIAOD'},
                              _MINI_from_MINI])
steps['HRV_NANO_mc'] = _HARVEST_mc
steps['HRV3_NANO_mc'] = merge([{'--filein':'file:step3_inDQM.root'},
                                 _HARVEST_mc])
steps['HRV_NANO_data'] = _HARVEST_data
steps['HRV3_NANO_data'] = merge([{'--filein':'file:step3_inDQM.root'},
                                 _HARVEST_data])

################################################################
# 10.6 INPUT and workflows
steps['TTbarAOD10.6_UL16APV'] = {'INPUT': InputInfo(
    location='STD', dataSet='/TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16RECOAPV-106X_mcRun2_asymptotic_preVFP_v8-v2/AODSIM')}
steps['MINI_mc10.6ul16APV'] = merge([{'--procModifiers':'run2_miniAOD_UL'}, steps['REMINIAOD_mc2016UL_preVFP']])
steps['M2M_mc10.6ul16APV'] = merge([_MINI_from_MINI, {'--procModifiers':'run2_miniAOD_miniAODUL'} ,steps['MINI_mc10.6ul16APV']])
steps['TTbarMINIAOD10.6_UL16APVv2'] = {'INPUT': InputInfo(
    location='STD', dataSet='/TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAODAPVv2-106X_mcRun2_asymptotic_preVFP_v11-v2/MINIAODSIM')}
steps['NANO_mc10.6ul16APVv2'] = merge([{'--era': 'Run2_2016_HIPM,run2_nanoAOD_106Xv2',
                                        '--conditions': 'auto:run2_mc'},
                                       _NANO_mc])
steps['TTbarAOD10.6_UL16'] = {'INPUT': InputInfo(
    location='STD', dataSet='/TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16RECO-106X_mcRun2_asymptotic_v13-v2/AODSIM')}
steps['MINI_mc10.6ul16'] = merge([{'--procModifiers':'run2_miniAOD_UL'}, steps['REMINIAOD_mc2016UL_postVFP']])
steps['M2M_mc10.6ul16'] = merge([_MINI_from_MINI, {'--procModifiers':'run2_miniAOD_miniAODUL'} ,steps['MINI_mc10.6ul16']])
steps['TTbarMINIAOD10.6_UL16v2'] = {'INPUT': InputInfo(
    location='STD', dataSet='/TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v2/MINIAODSIM')}
steps['NANO_mc10.6ul16v2'] = merge([{'--era': 'Run2_2016,run2_nanoAOD_106Xv2',
                                   '--conditions': 'auto:run2_mc'},
                                    _NANO_mc])
# 2017 looking Monte-Carlo: two versions in 10.6
steps['TTbarAOD10.6_UL17'] = {'INPUT': InputInfo(
    location='STD', dataSet='/TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17RECO-106X_mc2017_realistic_v6-v2/AODSIM')}
steps['MINI_mc10.6ul17'] = merge([{'--procModifiers':'run2_miniAOD_UL'}, steps['REMINIAOD_mc2017UL']])
steps['M2M_mc10.6ul17'] = merge([_MINI_from_MINI, {'--procModifiers':'run2_miniAOD_miniAODUL'} ,steps['MINI_mc10.6ul17']])
steps['TTbarMINIAOD10.6_UL17v2'] = {'INPUT': InputInfo(
    location='STD', dataSet='/TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v2/MINIAODSIM')}
steps['NANO_mc10.6ul17v2'] = merge([{'--era': 'Run2_2017,run2_nanoAOD_106Xv2',
                                   '--conditions': 'auto:phase1_2017_realistic'},
                                    _NANO_mc])

steps['TTbarAOD10.6_UL18'] = {'INPUT': InputInfo(
    location='STD', dataSet='/TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18RECO-106X_upgrade2018_realistic_v11_L1v1-v2/AODSIM')}
steps['MINI_mc10.6ul18'] = merge([{'--procModifiers':'run2_miniAOD_UL'}, steps['REMINIAOD_mc2018UL']])
steps['M2M_mc10.6ul18'] = merge([_MINI_from_MINI, {'--procModifiers':'run2_miniAOD_miniAODUL'} ,steps['MINI_mc10.6ul18']])
steps['TTbarMINIAOD10.6_UL18v2'] = {'INPUT': InputInfo(
    location='STD', dataSet='/TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM')}
steps['NANO_mc10.6ul18v2'] = merge([{'--era': 'Run2_2018,run2_nanoAOD_106Xv2',
                                   '--conditions': 'auto:phase1_2018_realistic'},
                                    _NANO_mc])

# HIPM_UL2016_MiniAODv2 campaign is CMSSW_10_6_25
steps['MuonEG2016MINIAOD10.6v2'] = {'INPUT': InputInfo(location='STD', ls=run2_lumis,
                                                       dataSet='/MuonEG/Run2016E-HIPM_UL2016_MiniAODv2-v2/MINIAOD')}
steps['M2M_data10.6ul16'] = merge([_MINI_from_MINI_data, {'--procModifiers':'run2_miniAOD_miniAODUL'}, steps['REMINIAOD_data2016UL']])
steps['NANO_data10.6ul16v2'] = merge([{'--era': 'Run2_2016_HIPM,run2_nanoAOD_106Xv2',
                                     '--conditions': 'auto:run2_data'},
                                      _NANO_data])
# UL2017_MiniAODv2 campaign is CMSSW_10_6_20
steps['MuonEG2017MINIAOD10.6v2'] = {'INPUT': InputInfo(location='STD', ls=run2_lumis,
                                                       dataSet='/MuonEG/Run2017F-UL2017_MiniAODv2-v1/MINIAOD')}
steps['M2M_data10.6ul17'] = merge([_MINI_from_MINI_data, {'--procModifiers':'run2_miniAOD_miniAODUL'}, steps['REMINIAOD_data2017UL']])
steps['NANO_data10.6ul17v2'] = merge([{'--era': 'Run2_2017,run2_nanoAOD_106Xv2',
                                     '--conditions': 'auto:run2_data'},
                                      _NANO_data])

# UL2018_MiniAODv2 campaign is CMSSW_10_6_20
steps['MuonEG2018MINIAOD10.6v2'] = {'INPUT': InputInfo(location='STD', ls=run2_lumis,
                                                       dataSet='/MuonEG/Run2018D-UL2018_MiniAODv2-v1/MINIAOD')}
steps['M2M_data10.6ul18'] = merge([_MINI_from_MINI_data, {'--procModifiers':'run2_miniAOD_miniAODUL'}, steps['REMINIAOD_data2018UL']])
steps['NANO_data10.6ul18v2'] = merge([{'--era': 'Run2_2018,run2_nanoAOD_106Xv2',
                                     '--conditions': 'auto:run2_data'},
                                      _NANO_data])

################################################################
# Run2UL re-MINI/NANO
## below: nano steps in current release, using MINI redone with the current release, either from AOD or MINIAOD
steps['NANO_mc_UL16APVreMINI'] = merge([{'--era': 'Run2_2016_HIPM',
                                         '--conditions': 'auto:run2_mc_pre_vfp'},
                                        _NANO_mc])
steps['NANO_mc_UL16reMINI'] = merge([{'--era': 'Run2_2016',
                                      '--conditions': 'auto:run2_mc'},
                                     _NANO_mc])
steps['NANO_mc_UL17reMINI'] = merge([{'--era': 'Run2_2017',
                                      '--conditions': 'auto:phase1_2017_realistic'},
                                     _NANO_mc])
steps['NANO_mc_UL18reMINI'] = merge([{'--era': 'Run2_2018',
                                      '--conditions': 'auto:phase1_2018_realistic'},
                                     _NANO_mc])

steps['NANO_data_UL16APVreMINI'] = merge([{'--era': 'Run2_2016_HIPM',
                                         '--conditions': 'auto:run2_data'},
                                          _NANO_data])
steps['NANO_data_UL16reMINI'] = merge([{'--era': 'Run2_2016',
                                      '--conditions': 'auto:run2_data'},
                                       _NANO_data])
steps['NANO_data_UL17reMINI'] = merge([{'--era': 'Run2_2017',
                                      '--conditions': 'auto:run2_data'},
                                       _NANO_data])
steps['NANO_data_UL18reMINI'] = merge([{'--era': 'Run2_2018',
                                      '--conditions': 'auto:run2_data'},
                                       _NANO_data])

################################################################
# 12.4 workflows -- data
steps['ScoutingPFRun3_Run2022D_RAW_124X'] = {'INPUT': InputInfo(
    dataSet='/ScoutingPFRun3/Run2022D-v1/RAW', label='2022D', events=100000, location='STD', ls=Run2022D)}

steps['ScoutingPFMonitor_Run2022D_RAW_124X'] = {'INPUT': InputInfo(
    dataSet='/ScoutingPFMonitor/Run2022D-v1/RAW', label='2022D', events=100000, location='STD', ls=Run2022D)}

steps['NANO_data12.4'] = merge([{'--era': 'Run3,run3_nanoAOD_pre142X', '--conditions': 'auto:run3_data'},
                                _NANO_data])

steps['scoutingNANO_data12.4'] = merge([{'-s': 'NANO:@Scout'},
                                        steps['NANO_data12.4']])

steps['scoutingNANO_monitor_data12.4'] = merge([{'-s': 'NANO:@ScoutMonitor'},
                                                steps['NANO_data12.4']])

################################################################
# 13.0 workflows
steps['TTbarMINIAOD13.0'] = {'INPUT': InputInfo(
    location='STD', dataSet='/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23MiniAODv4-130X_mcRun3_2023_realistic_v14-v2/MINIAODSIM')}
steps['M2M_mc13.0s23'] = merge([{'--era': 'Run3', '--conditions': 'auto:phase1_2023_realistic', '--procModifier': 'run3_miniAOD_miniAODpre142X'},
                             steps['M2M_mc10.6ul18']])
steps['NANO_mc13.0'] = merge([{'--era': 'Run3,run3_nanoAOD_pre142X', '--conditions': 'auto:phase1_2023_realistic'},
                              _NANO_mc])
## below: nano steps in current release, using MINI redone with the current release, either from AOD or MINIAOD
steps['NANO_mc_S23reMINI'] = merge([{'--era': 'Run3', '--conditions': 'auto:phase1_2023_realistic'},
                              _NANO_mc])
#steps['NANO_mc_S22reMINI'] = merge([{'--era': 'Run3', '--conditions': 'auto:phase1_2023_realistic'},## fig GT
#                              _NANO_mc])
#steps['NANO_mc_S22EEreMINI'] = merge([{'--era': 'Run3', '--conditions': 'auto:phase1_2023_realistic'},## fig GT
#                              _NANO_mc])
#steps['NANO_mc_S23reMINI'] = merge([{'--era': 'Run3_2023', '--conditions': 'auto:phase1_2023_realistic'},## fit GT
#                              _NANO_mc])
#steps['NANO_mc_S23BPixreMINI'] = merge([{'--era': 'Run3_2023', '--conditions': 'auto:phase1_2023_realistic'},## fix GT
#                              _NANO_mc])

# 13.0 workflows -- data
steps['MuonEG2023MINIAOD13.0'] = {'INPUT': InputInfo(location='STD', ls={368489: [[46, 546]]},
                                                     dataSet='/MuonEG/Run2023C-22Sep2023_v4-v1/MINIAOD')}

steps['ScoutingPFRun3_Run2023D_RAW_130X'] = {'INPUT': InputInfo(
    dataSet='/ScoutingPFRun3/Run2023D-v1/RAW', label='2023D', events=100000, location='STD', ls=Run2023D)}

steps['ScoutingPFMonitor_Run2023D_RAW_130X'] = {'INPUT': InputInfo(
    dataSet='/ScoutingPFMonitor/Run2023D-v1/RAW', label='2023D', events=100000, location='STD', ls=Run2023D)}

steps['M2M_data13.0'] = merge([_MINI_from_MINI_data, {'--era': 'Run3', '--conditions': 'auto:run3_data', '--procModifier': 'run3_miniAOD_miniAODpre142X'}])
steps['NANO_data13.0'] = merge([{'--era': 'Run3_2023,run3_nanoAOD_pre142X', '--conditions': 'auto:run3_data'},
                                _NANO_data])
steps['NANO_data_23reMINI'] = merge([{'--era': 'Run3_2023', '--conditions': 'auto:run3_data'},
                                  _NANO_data])
steps['NANO_data13.0_prompt'] = merge([{'-s': 'NANO:@Prompt,DQM:@nanoAODDQM', '-n': '1000'},
                                       steps['NANO_data13.0']])


steps['scoutingNANO_data13.0'] = merge([{'-s': 'NANO:@Scout'},
                                        steps['NANO_data13.0']])

steps['scoutingNANO_monitor_data13.0'] = merge([{'-s': 'NANO:@ScoutMonitor'},
                                                steps['NANO_data13.0']])


################################################################
# current release cycle workflows : 14.0
steps['TTbarMINIAOD14.0'] = {'INPUT': InputInfo(
    location='STD', dataSet='/RelValTTbar_14TeV/CMSSW_14_0_0-PU_140X_mcRun3_2024_realistic_v3_STD_2024_PU-v2/MINIAODSIM')}

steps['NANO_mc14.0'] = merge([{'--era': 'Run3,run3_nanoAOD_pre142X', '--conditions': 'auto:phase1_2024_realistic'},
                              _NANO_mc])

steps['muPOGNANO_mc14.0'] = merge([{'-s': 'NANO:@MUPOG,DQM:@nanoAODDQM', '-n': '1000'},
                                   steps['NANO_mc14.0']])

steps['EGMNANO_mc14.0'] = merge([{'-s': 'NANO:@EGM,DQM:@nanoAODDQM', '-n': '1000'},
                                 steps['NANO_mc14.0']])

steps['BTVNANO_mc14.0'] = merge([{'-s': 'NANO:@BTV,DQM:@nanoAODDQM', '-n': '1000'},
                                 steps['NANO_mc14.0']])

steps['lepTrackInfoNANO_mc14.0'] = merge([{'-s': 'NANO:@LepTrackInfo,DQM:@nanoAODDQM', '-n': '1000'},
                                          steps['NANO_mc14.0']])

steps['jmeNANO_mc14.0'] = merge([{'-s': 'NANO:@JME,DQM:@nanojmeDQM', '-n': '1000'},
                                 steps['NANO_mc14.0']])

steps['scoutingNANO_mc14.0'] = merge([{'-s': 'NANO:@Scout'},
                                      steps['NANO_mc14.0']])

steps['scoutingNANO_withPrompt_mc14.0'] = merge([{'-s': 'NANO:@Prompt+@Scout'},
                                                 steps['NANO_mc14.0']])

steps['SinglePionRAW14.0'] = {'INPUT': InputInfo(
    location='STD', dataSet='/SinglePion_E-50_Eta-0to3-pythia8-gun/Run3Winter25Digi-NoPU_142X_mcRun3_2025BOY_realistic_Candidate_2024_11_13_17_21_33-v2/GEN-SIM-RAW')}

steps['hcalDPGNANO_mc14.0'] = merge([{'-s': 'RAW2DIGI,RECO,NANO:@HCALMC', '-n': '100',
                                      '--processName': 'NANO'},
                                     steps['NANO_mc14.0']])

# 14.0 workflows -- data
lumis_Run2024D = {380306: [[28, 273]]}
steps['MuonEG2024MINIAOD14.0'] = {'INPUT': InputInfo(location='STD', ls=lumis_Run2024D,
                                                     dataSet='/MuonEG/Run2024D-PromptReco-v1/MINIAOD')}

steps['ScoutingPFRun32024RAW14.0'] = {'INPUT': InputInfo(location='STD', ls=lumis_Run2024D,
                                                         dataSet='/ScoutingPFRun3/Run2024D-v1/HLTSCOUT')}

steps['ScoutingPFMonitor2024MINIAOD14.0'] = {'INPUT': InputInfo(location='STD', ls=lumis_Run2024D,
                                                                dataSet='/ScoutingPFMonitor/Run2024D-PromptReco-v1/MINIAOD')}

steps['L1Scouting2024RAW14.0'] = {'INPUT': InputInfo(location='STD', ls={386873: [[1, 332]]},
                                                     dataSet='/L1Scouting/Run2024I-v1/L1SCOUT')}

steps['L1ScoutingSelection2024RAW14.0'] = {'INPUT': InputInfo(location='STD', ls={386873: [[1, 332]]},
                                                              dataSet='/L1ScoutingSelection/Run2024I-v1/L1SCOUT')}

steps['ZMuSkim2024RAWRECO14.0'] = {'INPUT': InputInfo(location='STD', ls=lumis_Run2024D,
                                                      dataSet='/Muon0/Run2024D-ZMu-PromptReco-v1/RAW-RECO')}

steps['ZeroBias2024RAW14.0'] = {'INPUT': InputInfo(location='STD', ls=lumis_Run2024D,
                                                   dataSet='/ZeroBias/Run2024D-v1/RAW')}

steps['TestEnablesEcalHcal2024RAW14.0'] = {'INPUT': InputInfo(location='STD', ls={383173: [[151, 162]]},
                                                              dataSet='/TestEnablesEcalHcal/Run2024F-Express-v1/RAW')}

steps['NANO_data14.0'] = merge([{'--era': 'Run3_2024,run3_nanoAOD_pre142X', '--conditions': 'auto:run3_data_prompt'},
                                _NANO_data])

steps['NANO_data14.0_prompt'] = merge([{'-s': 'NANO:@Prompt,DQM:@nanoAODDQM', '-n': '1000'},
                                       steps['NANO_data14.0']])

steps['muPOGNANO_data14.0'] = merge([{'-s': 'NANO:@MUPOG,DQM:@nanoAODDQM', '-n': '1000'},
                                     steps['NANO_data14.0']])

steps['EGMNANO_data14.0'] = merge([{'-s': 'NANO:@EGM,DQM:@nanoAODDQM', '-n': '1000'},
                                   steps['NANO_data14.0']])

steps['BTVNANO_data14.0'] = merge([{'-s': 'NANO:@BTV,DQM:@nanoAODDQM', '-n': '1000'},
                                   steps['NANO_data14.0']])

steps['lepTrackInfoNANO_data14.0'] = merge([{'-s': 'NANO:@LepTrackInfo,DQM:@nanoAODDQM', '-n': '1000'},
                                           steps['NANO_data14.0']])

steps['jmeNANO_data14.0'] = merge([{'-s': 'NANO:@JME,DQM:@nanojmeDQM', '-n': '1000'},
                                   steps['NANO_data14.0']])

steps['scoutingNANO_data14.0'] = merge([{'-s': 'NANO:@Scout'},
                                        steps['NANO_data14.0']])

steps['scoutingNANO_monitor_data14.0'] = merge([{'-s': 'NANO:@Prompt+@ScoutMonitor'},
                                                steps['NANO_data14.0']])

steps['scoutingNANO_monitorWithPrompt_data14.0'] = merge([{'-s': 'NANO:@Prompt+@ScoutMonitor'},
                                                           steps['NANO_data14.0']])

steps['l1ScoutingNANO_data14.0'] = merge([{'-s': 'NANO:@L1Scout', '-n': '1000'},
                                          steps['NANO_data14.0']])

steps['l1ScoutingSelectionNANO_data14.0'] = merge([{'-s': 'NANO:@L1ScoutSelect', '-n': '1000'},
                                                   steps['NANO_data14.0']])

# DPG custom NANO
steps['muDPGNANO_data14.0'] = merge([{'-s': 'RAW2DIGI,NANO:@MUDPG', '-n': '100'},
                                     steps['NANO_data14.0']])

steps['muDPGNANOBkg_data14.0'] = merge([{'-s': 'RAW2DIGI,RECO:localreco,NANO:@MUDPGBKG', '-n': '100'},
                                        steps['NANO_data14.0']])

steps['hcalDPGNANO_data14.0'] = merge([{'-s': 'RAW2DIGI,RECO,NANO:@HCAL', '-n': '100',
                                        '--processName': 'NANO'},
                                       steps['NANO_data14.0']])

steps['hcalDPGCalibNANO_data14.0'] = merge([{'-s': 'RAW2DIGI,RECO,NANO:@HCALCalib', '-n': '100',
                                             '--processName': 'NANO'},
                                            steps['NANO_data14.0']])

steps['l1DPGNANO_data14.0'] = merge([{'-s': 'RAW2DIGI,NANO:@L1DPG', '-n': '100'},
                                     steps['NANO_data14.0']])


################################################################
# Run3 re-MINI/NANOv15 in 15.0
steps['TTbar_13p6_Summer24_AOD_140X'] = {'INPUT': InputInfo(
    location='STD', dataSet='/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24DRPremix-140X_mcRun3_2024_realistic_v26-v2/AODSIM')}

steps['JetMET1_Run2024H_AOD_140X'] = {'INPUT': InputInfo(
    location='STD', ls={385836: [[72, 166]]}, dataSet='/JetMET1/Run2024H-PromptReco-v1/AOD')}

steps['NANO_mc_Summer24_reMINI'] = merge([{'--era': 'Run3_2024', '--conditions': 'auto:phase1_2024_realistic'}, _NANO_mc])

steps['NANO_data_2024_reMINI'] = merge([{'--era': 'Run3_2024', '--conditions': 'auto:run3_data'}, _NANO_data])


################################################################
# Run3, 15_0_X input (for 2025 data-taking)
# temporarily using the Summer24 samples
steps['TTbar_13p6_Summer24_MINIAOD_150X'] = {'INPUT': InputInfo(
    location='STD', dataSet='/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v2/MINIAODSIM')}

steps['NANO_mc15.0'] = merge([{'--era': 'Run3_2025', '--conditions': 'auto:phase1_2025_realistic'}, _NANO_mc])

steps['BPHNANO_mc15.0'] = merge([{'-s': 'NANO:@BPH,DQM:@nanoAODDQM'},
                                 steps['NANO_mc15.0']])

steps['muPOGNANO_mc15.0'] = merge([{'-s': 'NANO:@MUPOG,DQM:@nanoAODDQM', '-n': '1000'},
                                   steps['NANO_mc15.0']])

steps['EGMNANO_mc15.0'] = merge([{'-s': 'NANO:@EGM,DQM:@nanoAODDQM', '-n': '1000'},
                                 steps['NANO_mc15.0']])

steps['BTVNANO_mc15.0'] = merge([{'-s': 'NANO:@BTV,DQM:@nanoAODDQM', '-n': '1000'},
                                 steps['NANO_mc15.0']])

steps['EXONANO_mc15.0'] = merge([{'-s': 'PAT,NANO:@EXO', '-n': '1000'},
                                 steps['NANO_mc15.0']])

steps['lepTrackInfoNANO_mc15.0'] = merge([{'-s': 'NANO:@LepTrackInfo,DQM:@nanoAODDQM', '-n': '1000'},
                                          steps['NANO_mc15.0']])

steps['jmeNANO_mc15.0'] = merge([{'-s': 'NANO:@JME,DQM:@nanojmeDQM', '-n': '1000'},
                                 steps['NANO_mc15.0']])

steps['jmeNANO_rePuppi_mc15.0'] = merge([{'-s': 'NANO:@JMErePuppi,DQM:@nanojmeDQM', '-n': '1000'},
                                         steps['NANO_mc15.0']])

steps['scoutingNANO_mc15.0'] = merge([{'-s': 'NANO:@Scout'},
                                      steps['NANO_mc15.0']])

steps['scoutingNANO_withPrompt_mc15.0'] = merge([{'-s': 'NANO:@Prompt+@Scout'},
                                                 steps['NANO_mc15.0']])

# ====== DATA ======
lumis_Run2025C = {392175: [[95, 542]]}
lumis_Run2025C_Muon0 = {392997: [[146, 202]]}

steps['JetMET1_Run2025C_MINIAOD_150X'] = {'INPUT': InputInfo(
    location='STD', ls=lumis_Run2025C, dataSet='/JetMET1/Run2025C-PromptReco-v1/MINIAOD')}

steps['Muon0_Run2025C_AOD_150X'] = {'INPUT': InputInfo(
    location='STD',ls=lumis_Run2025C_Muon0, dataSet='/Muon0/Run2025C-PromptReco-v1/AOD')}

steps['ScoutingPFRun3_Run2025C_HLTSCOUT_150X'] = {'INPUT': InputInfo(location='STD', ls=lumis_Run2025C,
                                                         dataSet='/ScoutingPFRun3/Run2025C-v1/HLTSCOUT')}

steps['ScoutingPFMonitor_Run2025C_MINIAOD_150X'] = {'INPUT': InputInfo(
    location='STD', ls=lumis_Run2025C, dataSet='/ScoutingPFMonitor/Run2025C-PromptReco-v1/MINIAOD')}

steps['NANO_data15.0'] = merge([{'--era': 'Run3_2025', '--conditions': 'auto:run3_data_prompt'}, _NANO_data])

steps['NANO_data15.0_prompt'] = merge([{'-s': 'NANO:@Prompt,DQM:@nanoAODDQM', '-n': '1000'},
                                       steps['NANO_data15.0']])

steps['BPHNANO_data15.0'] = merge([{'-s': 'NANO:@BPH,DQM:@nanoAODDQM'},
                                   steps['NANO_data15.0']])

steps['muPOGNANO_data15.0'] = merge([{'-s': 'NANO:@MUPOG,DQM:@nanoAODDQM', '-n': '1000'},
                                     steps['NANO_data15.0']])

steps['EGMNANO_data15.0'] = merge([{'-s': 'NANO:@EGM,DQM:@nanoAODDQM', '-n': '1000'},
                                   steps['NANO_data15.0']])

steps['BTVNANO_data15.0'] = merge([{'-s': 'NANO:@BTV,DQM:@nanoAODDQM', '-n': '1000'},
                                   steps['NANO_data15.0']])

steps['EXONANO_data15.0'] = merge([{'-s': 'PAT,NANO:@EXO', '-n': '1000'},
                                   steps['NANO_data15.0']])

steps['lepTrackInfoNANO_data15.0'] = merge([{'-s': 'NANO:@LepTrackInfo,DQM:@nanoAODDQM', '-n': '1000'},
                                           steps['NANO_data15.0']])

steps['jmeNANO_data15.0'] = merge([{'-s': 'NANO:@JME,DQM:@nanojmeDQM', '-n': '1000'},
                                   steps['NANO_data15.0']])

steps['jmeNANO_rePuppi_data15.0'] = merge([{'-s': 'NANO:@JMErePuppi,DQM:@nanojmeDQM', '-n': '1000'},
                                          steps['NANO_data15.0']])

steps['scoutingNANO_data15.0'] = merge([{'-s': 'NANO:@Scout'},
                                        steps['NANO_data15.0']])

steps['scoutingNANO_monitor_data15.0'] = merge([{'-s': 'NANO:@ScoutMonitor'},
                                                steps['NANO_data15.0']])

steps['scoutingNANO_monitorWithPrompt_data15.0'] = merge([{'-s': 'NANO:@Prompt+@ScoutMonitor'},
                                                          steps['NANO_data15.0']])

################################################################
# NANOGEN
steps['NANOGENFromGen'] = merge([{'-s': 'NANO:@GEN,DQM:@nanogenDQM',
                                  '-n': 1000,
                                  '--conditions': 'auto:run2_mc'},
                                 _NANO_mc])
steps['NANOGENFromMini'] = merge([{'-s': 'NANO:@GENFromMini,DQM:@nanogenDQM',
                                   '-n': 1000,
                                   '--conditions': 'auto:run2_mc'},
                                  _NANO_mc])

################################################################
_wfn = WFN(2500)
######## 2500.0xxx ########
# Run2, 10_6_X MiniAOD input (current recommendation for 2016--2018)
workflows[_wfn()] = ['NANOmc106Xul16v2', ['TTbarMINIAOD10.6_UL16v2', 'NANO_mc10.6ul16v2', 'HRV_NANO_mc']]
workflows[_wfn()] = ['NANOmc106Xul16APVv2', ['TTbarMINIAOD10.6_UL16APVv2', 'NANO_mc10.6ul16APVv2', 'HRV_NANO_mc']]
workflows[_wfn()] = ['NANOmc106Xul17v2', ['TTbarMINIAOD10.6_UL17v2', 'NANO_mc10.6ul17v2', 'HRV_NANO_mc']]
workflows[_wfn()] = ['NANOmc106Xul18v2', ['TTbarMINIAOD10.6_UL18v2', 'NANO_mc10.6ul18v2', 'HRV_NANO_mc']]

_wfn.subnext()
workflows[_wfn()] = ['NANOdata106Xul16v2', ['MuonEG2016MINIAOD10.6v2', 'NANO_data10.6ul16v2', 'HRV_NANO_data']]
workflows[_wfn()] = ['NANOdata106Xul17v2', ['MuonEG2017MINIAOD10.6v2', 'NANO_data10.6ul17v2', 'HRV_NANO_data']]
workflows[_wfn()] = ['NANOdata106Xul18v2', ['MuonEG2018MINIAOD10.6v2', 'NANO_data10.6ul18v2', 'HRV_NANO_data']]

# Run2, 10_6_X AOD, reMINI+reNANO
_wfn.subnext()
workflows[_wfn()] = ['NANOmcUL16APVreMINI', ['TTbarAOD10.6_UL16APV', 'MINI_mc10.6ul16APV', 'NANO_mc_UL16APVreMINI', 'HRV3_NANO_mc']]
workflows[_wfn()] = ['NANOmcUL16reMINI', ['TTbarAOD10.6_UL16', 'MINI_mc10.6ul16', 'NANO_mc_UL16reMINI', 'HRV3_NANO_mc']]
workflows[_wfn()] = ['NANOmcUL17reMINI', ['TTbarAOD10.6_UL17', 'MINI_mc10.6ul17', 'NANO_mc_UL17reMINI', 'HRV3_NANO_mc']]
workflows[_wfn()] = ['NANOmcUL18reMINI', ['TTbarAOD10.6_UL18', 'MINI_mc10.6ul18', 'NANO_mc_UL18reMINI', 'HRV3_NANO_mc']]

_wfn.subnext()
workflows[_wfn()] = ['NANOdataUL16APVreMINI', ['RunJetHT2016E_reminiaodUL', 'REMINIAOD_data2016UL_HIPM', 'NANO_data_UL16APVreMINI', 'HRV3_NANO_data']]  # noqa
workflows[_wfn()] = ['NANOdataUL16reMINI', ['RunJetHT2016H_reminiaodUL', 'REMINIAOD_data2016UL', 'NANO_data_UL16reMINI', 'HRV3_NANO_data']]  # noqa
workflows[_wfn()] = ['NANOdataUL17reMINI', ['RunJetHT2017F_reminiaodUL', 'REMINIAOD_data2017UL', 'NANO_data_UL17reMINI', 'HRV3_NANO_data']]  # noqa
workflows[_wfn()] = ['NANOdataUL18reMINI', ['RunJetHT2018D_reminiaodUL', 'REMINIAOD_data2018UL', 'NANO_data_UL18reMINI', 'HRV3_NANO_data']]  # noqa

_wfn.subnext()
workflows[_wfn()] = ['NANOmcUL16APVMini2Mini', ['TTbarMINIAOD10.6_UL16APVv2', 'M2M_mc10.6ul16APV', 'NANO_mc_UL16APVreMINI', 'HRV3_NANO_mc']]
workflows[_wfn()] = ['NANOmcUL16Mini2Mini', ['TTbarMINIAOD10.6_UL16v2', 'M2M_mc10.6ul16', 'NANO_mc_UL16reMINI', 'HRV3_NANO_mc']]
workflows[_wfn()] = ['NANOmcUL17Mini2Mini', ['TTbarMINIAOD10.6_UL17v2', 'M2M_mc10.6ul17', 'NANO_mc_UL17reMINI', 'HRV3_NANO_mc']]
workflows[_wfn()] = ['NANOmcUL18Mini2Mini', ['TTbarMINIAOD10.6_UL18v2', 'M2M_mc10.6ul18', 'NANO_mc_UL18reMINI', 'HRV3_NANO_mc']]

_wfn.subnext()
workflows[_wfn()] = ['NANOdataUL16Mini2Mini', ['MuonEG2016MINIAOD10.6v2', 'M2M_data10.6ul16', 'NANO_data_UL16reMINI', 'HRV3_NANO_data']]
workflows[_wfn()] = ['NANOdataUL17Mini2Mini', ['MuonEG2017MINIAOD10.6v2', 'M2M_data10.6ul17', 'NANO_data_UL17reMINI', 'HRV3_NANO_data']]
workflows[_wfn()] = ['NANOdataUL18Mini2Mini', ['MuonEG2018MINIAOD10.6v2', 'M2M_data10.6ul18', 'NANO_data_UL18reMINI', 'HRV3_NANO_data']]

_wfn.next(1)
######## 2500.1xxx ########
# Run3, 13_0_X input (current recommendation for 2022--2023)
workflows[_wfn()] = ['NANOmc130X', ['TTbarMINIAOD13.0', 'NANO_mc13.0', 'HRV_NANO_mc']]

_wfn.subnext()
workflows[_wfn()] = ['NANOdata130Xrun3', ['MuonEG2023MINIAOD13.0', 'NANO_data13.0', 'HRV_NANO_data']]

_wfn.subnext()
workflows[_wfn()] = ['NANOmc23Mini2Mini', ['TTbarMINIAOD13.0', 'M2M_mc13.0s23', 'NANO_mc_S23reMINI', 'HRV3_NANO_mc']]

_wfn.subnext()
workflows[_wfn()] = ['NANOdata23Mini2Mini', ['MuonEG2023MINIAOD13.0', 'M2M_data13.0', 'NANO_data_23reMINI', 'HRV3_NANO_data']]

# POG/PAG custom NANOs, MC
_wfn.subnext()

# POG/PAG custom NANOs, data
_wfn.subnext()
workflows[_wfn()] = ['ScoutingNANOdata124Xrun3', ['ScoutingPFRun3_Run2022D_RAW_124X', 'scoutingNANO_data12.4']]
workflows[_wfn()] = ['ScoutingNANOmonitordata124Xrun3', ['ScoutingPFMonitor_Run2022D_RAW_124X', 'scoutingNANO_monitor_data12.4']]
workflows[_wfn()] = ['ScoutingNANOdata130Xrun3', ['ScoutingPFRun3_Run2023D_RAW_130X', 'scoutingNANO_data13.0']]
workflows[_wfn()] = ['ScoutingNANOmonitordata130Xrun3', ['ScoutingPFMonitor_Run2023D_RAW_130X', 'scoutingNANO_monitor_data13.0']]

# DPG custom NANOs, data
_wfn.subnext()

_wfn.next(2)
######## 2500.2xxx ########
# Run3, 14_0_X input (2024 RAW/AOD)
# Standard NANO, MC

# Standard NANO, data
_wfn.subnext()

# POG/PAG custom NANOs, MC
_wfn.subnext()
workflows[_wfn()] = ['muPOGNANOmc140X', ['TTbarMINIAOD14.0', 'muPOGNANO_mc14.0']]
workflows[_wfn()] = ['EGMNANOmc140X', ['TTbarMINIAOD14.0', 'EGMNANO_mc14.0']]
workflows[_wfn()] = ['BTVNANOmc140X', ['TTbarMINIAOD14.0', 'BTVNANO_mc14.0']]
workflows[_wfn()] = ['jmeNANOmc140X', ['TTbarMINIAOD14.0', 'jmeNANO_mc14.0']]
_wfn() # workflows[_wfn()] = ['jmeNANOrePuppimc140X', ['TTbarMINIAOD14.0', 'jmeNANO_rePuppi_mc14.0']]
workflows[_wfn()] = ['lepTrackInfoNANOmc140X', ['TTbarMINIAOD14.0', 'lepTrackInfoNANO_mc14.0']]
workflows[_wfn()] = ['ScoutingNANOmc140X', ['TTbarMINIAOD14.0', 'scoutingNANO_mc14.0']]
workflows[_wfn()] = ['ScoutingNANOwithPromptmc140X', ['TTbarMINIAOD14.0', 'scoutingNANO_withPrompt_mc14.0']]

# POG/PAG custom NANOs, data
_wfn.subnext()
workflows[_wfn()] = ['muPOGNANO140Xrun3', ['MuonEG2024MINIAOD14.0', 'muPOGNANO_data14.0']]
workflows[_wfn()] = ['EGMNANOdata140Xrun3', ['MuonEG2024MINIAOD14.0', 'EGMNANO_data14.0']]
workflows[_wfn()] = ['BTVNANOdata140Xrun3', ['MuonEG2024MINIAOD14.0', 'BTVNANO_data14.0']]
workflows[_wfn()] = ['jmeNANOdata140Xrun3', ['MuonEG2024MINIAOD14.0', 'jmeNANO_data14.0']]
_wfn()  # workflows[_wfn()] = ['jmeNANOrePuppidata140Xrun3', ['MuonEG2024MINIAOD14.0', 'jmeNANO_rePuppi_data14.0']]
workflows[_wfn()] = ['lepTrackInfoNANOdata140Xrun3', ['MuonEG2024MINIAOD14.0', 'lepTrackInfoNANO_data14.0']]
workflows[_wfn()] = ['ScoutingNANOdata140Xrun3', ['ScoutingPFRun32024RAW14.0', 'scoutingNANO_data14.0']]
workflows[_wfn()] = ['ScoutingNANOmonitordata140Xrun3', ['ScoutingPFMonitor2024MINIAOD14.0', 'scoutingNANO_monitor_data14.0']]
workflows[_wfn()] = ['ScoutingNANOmonitorWithPromptdata140Xrun3', ['ScoutingPFMonitor2024MINIAOD14.0', 'scoutingNANO_monitorWithPrompt_data14.0']]
workflows[_wfn()] = ['L1ScoutingNANOdata140Xrun3', ['L1Scouting2024RAW14.0', 'l1ScoutingNANO_data14.0']]
workflows[_wfn()] = ['L1ScoutingSelectionNANOdata140Xrun3', ['L1ScoutingSelection2024RAW14.0', 'l1ScoutingSelectionNANO_data14.0']]

# DPG custom NANOs, data
_wfn.subnext()
workflows[_wfn()] = ['l1DPGNANO140Xrun3', ['ZMuSkim2024RAWRECO14.0', 'l1DPGNANO_data14.0']]
workflows[_wfn()] = ['muDPGNANO140Xrun3', ['ZMuSkim2024RAWRECO14.0', 'muDPGNANO_data14.0']]
workflows[_wfn()] = ['muDPGNANOBkg140Xrun3', ['ZeroBias2024RAW14.0', 'muDPGNANOBkg_data14.0']]
workflows[_wfn()] = ['hcalDPGNANO140Xrun3', ['ZeroBias2024RAW14.0', 'hcalDPGNANO_data14.0']]
workflows[_wfn()] = ['hcalDPGCalibNANO140Xrun3', ['TestEnablesEcalHcal2024RAW14.0', 'hcalDPGCalibNANO_data14.0']]

# DPG custom NANOs, MC
_wfn.subnext()
workflows[_wfn()] = ['hcalDPGMCNANO140Xrun3', ['SinglePionRAW14.0', 'hcalDPGNANO_mc14.0']]
# The above HCAL workflow is actually using data produced for 14.2
# but I keep the 14.0 label for now since it's consistent with those ones
# let me know if I should change this

# MINIv6+NANOv15, MC
_wfn.subnext()
workflows[_wfn()] = ['NANOmc2024reMINI', ['TTbar_13p6_Summer24_AOD_140X', 'REMINIAOD_mc2024', 'NANO_mc_Summer24_reMINI', 'HRV3_NANO_mc']]  # noqa

# MINIv6+NANOv15, data
_wfn.subnext()
workflows[_wfn()] = ['NANOdata2024reMINI', ['JetMET1_Run2024H_AOD_140X', 'REMINIAOD_data2024', 'NANO_data_2024_reMINI', 'HRV3_NANO_data']]  # noqa

_wfn.next(3)
######## 2500.3xxx ########
# Run3, 15_0_X input (2024 MINIv6+NANOv15 & 2025 data-taking)
# Standard NANO, MC
workflows[_wfn()] = ['NANOmc150X', ['TTbar_13p6_Summer24_MINIAOD_150X', 'NANO_mc15.0', 'HRV_NANO_mc']]

# Standard NANO, data
_wfn.subnext()
workflows[_wfn()] = ['NANOdata150X', ['JetMET1_Run2025C_MINIAOD_150X', 'NANO_data15.0', 'HRV_NANO_data']]
workflows[_wfn()] = ['NANOdata150X', ['JetMET1_Run2025C_MINIAOD_150X', 'NANO_data15.0_prompt', 'HRV_NANO_data']]

# POG/PAG custom NANOs, MC
_wfn.subnext()
workflows[_wfn()] = ['muPOGNANOmc150X', ['TTbar_13p6_Summer24_MINIAOD_150X', 'muPOGNANO_mc15.0']]
workflows[_wfn()] = ['EGMNANOmc150X', ['TTbar_13p6_Summer24_MINIAOD_150X', 'EGMNANO_mc15.0']]
workflows[_wfn()] = ['BTVNANOmc150X', ['TTbar_13p6_Summer24_MINIAOD_150X', 'BTVNANO_mc15.0']]
workflows[_wfn()] = ['jmeNANOmc150X', ['TTbar_13p6_Summer24_MINIAOD_150X', 'jmeNANO_mc15.0']]
workflows[_wfn()] = ['jmeNANOrePuppimc150X', ['TTbar_13p6_Summer24_MINIAOD_150X', 'jmeNANO_rePuppi_mc15.0']]
workflows[_wfn()] = ['lepTrackInfoNANOmc150X', ['TTbar_13p6_Summer24_MINIAOD_150X', 'lepTrackInfoNANO_mc15.0']]
workflows[_wfn()] = ['ScoutingNANOmc150X', ['TTbar_13p6_Summer24_MINIAOD_150X', 'scoutingNANO_mc15.0']]
workflows[_wfn()] = ['ScoutingNANOwithPromptmc150X', ['TTbar_13p6_Summer24_MINIAOD_150X', 'scoutingNANO_withPrompt_mc15.0']]
workflows[_wfn()] = ['BPHNANOmc150X', ['TTbar_13p6_Summer24_MINIAOD_150X', 'BPHNANO_mc15.0']]
workflows[_wfn()] = ['EXONANOmc150X', ['TTbar_13p6_Summer24_AOD_140X', 'EXONANO_mc15.0']]

# POG/PAG custom NANOs, data
_wfn.subnext()
workflows[_wfn()] = ['muPOGNANO150Xrun3', ['JetMET1_Run2025C_MINIAOD_150X', 'muPOGNANO_data15.0']]
workflows[_wfn()] = ['EGMNANOdata150Xrun3', ['JetMET1_Run2025C_MINIAOD_150X', 'EGMNANO_data15.0']]
workflows[_wfn()] = ['BTVNANOdata150Xrun3', ['JetMET1_Run2025C_MINIAOD_150X', 'BTVNANO_data15.0']]
workflows[_wfn()] = ['jmeNANOdata150Xrun3', ['JetMET1_Run2025C_MINIAOD_150X', 'jmeNANO_data15.0']]
workflows[_wfn()] = ['jmeNANOrePuppidata150Xrun3', ['JetMET1_Run2025C_MINIAOD_150X', 'jmeNANO_rePuppi_data15.0']]
workflows[_wfn()] = ['lepTrackInfoNANOdata150Xrun3', ['JetMET1_Run2025C_MINIAOD_150X', 'lepTrackInfoNANO_data15.0']]
workflows[_wfn()] = ['ScoutingNANOdata150Xrun3', ['ScoutingPFRun3_Run2025C_HLTSCOUT_150X', 'scoutingNANO_data15.0']]
workflows[_wfn()] = ['ScoutingNANOmonitordata150Xrun3', ['ScoutingPFMonitor_Run2025C_MINIAOD_150X', 'scoutingNANO_monitor_data15.0']]  # noqa
workflows[_wfn()] = ['ScoutingNANOmonitorWithPromptdata150Xrun3', ['ScoutingPFMonitor_Run2025C_MINIAOD_150X', 'scoutingNANO_monitorWithPrompt_data15.0']]  # noqa
workflows[_wfn()] = ['BPHNANOdata150Xrun3', ['JetMET1_Run2025C_MINIAOD_150X', 'BPHNANO_data15.0']]
workflows[_wfn()] = ['EXONANOdata150Xrun3', ['Muon0_Run2025C_AOD_150X', 'EXONANO_data15.0']]

# DPG custom NANOs, data
_wfn.subnext()

# DPG custom NANOs, MC
_wfn.subnext()


_wfn.next(9)
######## 2500.9xxx ########
# NANOGEN
workflows[_wfn()] = ['', ['TTbarMINIAOD10.6_UL18v2', 'NANOGENFromMini']]
workflows[_wfn()] = ['', ['TTbarMINIAOD14.0', 'NANOGENFromMini']]
_wfn.subnext()
workflows[_wfn()] = ['', ['DYToLL_M-50_13TeV_pythia8', 'NANOGENFromGen']]
workflows[_wfn()] = ['', ['DYToll01234Jets_5f_LO_MLM_Madgraph_LHE_13TeV',
                          'Hadronizer_TuneCP5_13TeV_MLM_5f_max4j_LHE_pythia8', 'NANOGENFromGen']]
workflows[_wfn()] = ['', ['TTbar_Pow_LHE_13TeV', 'Hadronizer_TuneCP5_13TeV_powhegEmissionVeto2p_pythia8', 'NANOGENFromGen']]
