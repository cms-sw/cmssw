from  Configuration.PyReleaseValidation.relval_steps import *
import math

workflows = Matrix()

_runOnly20events={'-n':'20'}
_run10kevents={'-n':'10000'}

_NANO_data = merge([{'-s':'NANO,DQM:@nanoAODDQM',
                     '--process':'NANO',
                     '--data':'',
                     '--eventcontent':'NANOAOD,DQM',
                     '--datatier':'NANOAOD,DQMIO',
                     '-n':'10000',
                     '--customise':'"Configuration/DataProcessing/Utils.addMonitoring"'
                 }])
_HARVEST_nano = merge([{'-s':'HARVESTING:@nanoAODDQM',
                        '--filetype':'DQM',
                        '--filein':'file:step2_inDQM.root',
                        '--conditions':'auto:run2_data'## this is fake for harvesting
                    }])
_HARVEST_data = merge([_HARVEST_nano, {'--data':''}])


run2_lumis={ 277168: [[1, 1708]],
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
run3_lumis={}

_NANO_mc = merge([{'-s':'NANO,DQM:@nanoAODDQM',
                   '--process':'NANO',
                   '--mc':'',
                   '--eventcontent':'NANOAODSIM,DQM',
                   '--datatier':'NANOAODSIM,DQMIO',
                   '-n':'10000',
                   '--customise':'"Configuration/DataProcessing/Utils.addMonitoring"'
               }])
_HARVEST_mc = merge([_HARVEST_nano, {'--mc':''}])
steps['HRV_NANO_mc'] = _HARVEST_mc
steps['HRV_NANO_data'] = _HARVEST_data

##10.6 INPUT and worflows
steps['TTbarMINIAOD10.6_UL16v2'] = {'INPUT':InputInfo(location='STD',
                                                      dataSet='/TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v2/MINIAODSIM')}
steps['NANO_mc10.6ul16v2']=merge([{'--era':'Run2_2016,run2_nanoAOD_106Xv2',
                                   '--conditions':'auto:run2_mc'},
                                  _NANO_mc])
##2017 looking Monte-Carlo: two versions in 10.6
steps['TTbarMINIAOD10.6_UL17v2'] = {'INPUT':InputInfo(location='STD',
                                                      dataSet='/TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v2/MINIAODSIM')}
steps['NANO_mc10.6ul17v2']=merge([{'--era':'Run2_2017,run2_nanoAOD_106Xv2',
                                   '--conditions':'auto:phase1_2017_realistic'},
                                  _NANO_mc])

steps['TTbarMINIAOD10.6_UL18v2'] = {'INPUT':InputInfo(location='STD',
                                                      dataSet='/TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM')}
steps['NANO_mc10.6ul18v2']=merge([{'--era':'Run2_2018,run2_nanoAOD_106Xv2',
                                   '--conditions':'auto:phase1_2018_realistic'},
                                  _NANO_mc])

##HIPM_UL2016_MiniAODv2 campaign is CMSSW_10_6_25
steps['MuonEG2016MINIAOD10.6v2'] = {'INPUT':InputInfo(location='STD',ls=run2_lumis,
                                                      dataSet='/MuonEG/Run2016E-HIPM_UL2016_MiniAODv2-v2/MINIAOD')}
steps['NANO_data10.6ul16v2']=merge([{'--era':'Run2_2016,run2_nanoAOD_106Xv2,tracker_apv_vfp30_2016',
                                     '--conditions':'auto:run2_data'},
                                    _NANO_data])
##UL2017_MiniAODv2 campaign is CMSSW_10_6_20
steps['MuonEG2017MINIAOD10.6v2'] = {'INPUT':InputInfo(location='STD',ls=run2_lumis,
                                                      dataSet='/MuonEG/Run2017F-UL2017_MiniAODv2-v1/MINIAOD')}
steps['NANO_data10.6ul17v2']=merge([{'--era':'Run2_2017,run2_nanoAOD_106Xv2',
                                     '--conditions':'auto:run2_data'},
                                    _NANO_data])

##UL2018_MiniAODv2 campaign is CMSSW_10_6_20
steps['MuonEG2018MINIAOD10.6v2'] = {'INPUT':InputInfo(location='STD',ls=run2_lumis,
                                                      dataSet='/MuonEG/Run2018D-UL2018_MiniAODv2-v1/MINIAOD')}
steps['NANO_data10.6ul18v2']=merge([{'--era':'Run2_2018,run2_nanoAOD_106Xv2',
                                     '--conditions':'auto:run2_data'},
                                    _NANO_data])
##12.2 INPUT (mc only)
steps['TTbarMINIAOD12.2'] = {'INPUT':InputInfo(location='STD',
                                               dataSet='/TTToSemiLeptonic_TuneCP5_13p6TeV-powheg-pythia8/Run3Winter22MiniAOD-FlatPU0to70_122X_mcRun3_2021_realistic_v9-v2/MINIAODSIM')}
steps['NANO_mc12.2']=merge([{'--era':'Run3,run3_nanoAOD_122',
                             '--conditions':'auto:phase1_2022_realistic'},
                            _NANO_mc])

##12.4 INPUT
steps['TTbarMINIAOD12.4'] = {'INPUT':InputInfo(location='STD',
                                               dataSet='/TT_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22MiniAODv3-124X_mcRun3_2022_realistic_v12-v3/MINIAODSIM')}
steps['NANO_mc12.4']=merge([{'--era':'Run3,run3_nanoAOD_124',
                             '--conditions':'124X_mcRun3_2022_realistic_v12'},
                            _NANO_mc])

steps['MuonEG2022MINIAOD12.4'] = {'INPUT':InputInfo(location='STD',ls=run3_lumis,
                                                    dataSet='/MuonEG/Run2022D-PromptReco-v2/MINIAOD')}
steps['NANO_data12.4']=merge([{'--era':'Run3,run3_nanoAOD_124',
                               '--conditions':'auto:run3_data'},
                              _NANO_data])
steps['NANO_data12.4_prompt']=merge([{'--customise' : 'PhysicsTools/NanoAOD/nano_cff.nanoL1TrigObjCustomize', '-n' : '1000'},
                                     steps['NANO_data12.4']])

###13.0 workflows
steps['TTBarMINIAOD13.0'] = {'INPUT':InputInfo(location='STD',
                                               dataSet='/RelValTTbar_14TeV/CMSSW_13_0_0-PU_130X_mcRun3_2022_realistic_v2_HS-v4/MINIAODSIM')}

steps['NANO_mc13.0']=merge([{'--era':'Run3',
                             '--conditions':'130X_mcRun3_2022_realistic_v2'},
                            _NANO_mc])

steps['MuonEG2023MINIAOD13.0'] = { 'INPUT':InputInfo(location='STD',ls=run3_lumis,
                                                     dataSet='/MuonEG/Run2023C-PromptReco-v4/MINIAOD')}

steps['NANO_data13.0']=merge([{'--era':'Run3',
                               '--conditions':'auto:run3_data'},
                              _NANO_data])

steps['NANO_data13.0_prompt']=merge([{'--customise' : 'PhysicsTools/NanoAOD/nano_cff.nanoL1TrigObjCustomize', '-n' : '1000'},
                                     steps['NANO_data13.0']])

###current release cycle workflows : 13.2
steps['TTBarMINIAOD13.2'] = {'INPUT':InputInfo(location='STD',
                                               ## dataset below to be replaced with a 13.2 relval sample when available
                                               dataSet='/RelValTTbar_14TeV/CMSSW_13_0_0-PU_130X_mcRun3_2022_realistic_v2_HS-v4/MINIAODSIM')}

steps['NANO_mc13.2']=merge([{'--era':'Run3',
                             '--conditions':'auto:phase1_2022_realistic'},
                            _NANO_mc])

################
#10.6 input
workflows[2500.100] = ['NANOmc106Xul16v2', ['TTbarMINIAOD10.6_UL16v2','NANO_mc10.6ul16v2', 'HRV_NANO_mc']]
workflows[2500.101] = ['NANOmc106Xul17v2', ['TTbarMINIAOD10.6_UL17v2','NANO_mc10.6ul17v2', 'HRV_NANO_mc']]
workflows[2500.102] = ['NANOmc106Xul18v2', ['TTbarMINIAOD10.6_UL18v2','NANO_mc10.6ul18v2', 'HRV_NANO_mc']]

workflows[2500.110] = ['NANOdata106Xul16v2', ['MuonEG2016MINIAOD10.6v2', 'NANO_data10.6ul16v2', 'HRV_NANO_data']]
workflows[2500.111] = ['NANOdata106Xul17v2', ['MuonEG2017MINIAOD10.6v2', 'NANO_data10.6ul17v2', 'HRV_NANO_data']]
workflows[2500.112] = ['NANOdata106Xul18v2', ['MuonEG2018MINIAOD10.6v2', 'NANO_data10.6ul18v2', 'HRV_NANO_data']]

################
#12.2 input
workflows[2500.200] = ['NANOmc122Xrun3', ['TTbarMINIAOD12.2','NANO_mc12.2', 'HRV_NANO_mc']]

################
#12.4 input
workflows[2500.300] = ['NANOmc124Xrun3', ['TTbarMINIAOD12.4','NANO_mc12.4', 'HRV_NANO_mc']]

workflows[2500.310] = ['NANOdata124Xrun3', ['MuonEG2022MINIAOD12.4','NANO_data12.4', 'HRV_NANO_data']]
workflows[2500.311] = ['NANOdata124Xrun3', ['MuonEG2022MINIAOD12.4','NANO_data12.4_prompt', 'HRV_NANO_data']]

################
#13.0 workflows
workflows[2500.400] = ['NANOmc130X', ['TTBarMINIAOD13.0', 'NANO_mc13.0', 'HRV_NANO_mc']]

workflows[2500.410] = ['NANOdata130Xrun3', ['MuonEG2023MINIAOD13.0', 'NANO_data13.0', 'HRV_NANO_data']]
workflows[2500.411] = ['NANOdata130Xrun3', ['MuonEG2023MINIAOD13.0', 'NANO_data13.0_prompt', 'HRV_NANO_data']]

################
#13.2 workflows
workflows[2500.500] = ['NANOmc132X', ['TTBarMINIAOD13.2', 'NANO_mc13.2', 'HRV_NANO_mc']]
