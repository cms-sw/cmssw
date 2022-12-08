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
                                               ## to be updated as soon as some TTbar appears in a 12.4 campaign
                                               dataSet='/RelValTTbar_14TeV/CMSSW_12_4_9_patch1-124X_mcRun3_2022_realistic_v10_BS2022-v1/MINIAODSIM')}
steps['NANO_mc12.4']=merge([{'--era':'Run3,run3_nanoAOD_124',
                             '--conditions':'auto:phase1_2022_realistic'},
                            _NANO_mc])

steps['MuonEG2022MINIAOD12.4'] = {'INPUT':InputInfo(location='STD',ls=run3_lumis,
                                                    dataSet='/MuonEG/Run2022D-PromptReco-v2/MINIAOD')}
steps['NANO_data12.4']=merge([{'--era':'Run3,run3_nanoAOD_124',
                               '--conditions':'auto:run3_data'},
                              _NANO_data])

##12.6 workflows
steps['TTBarMINIAOD12.6'] = {'INPUT':InputInfo(location='STD',ls=run3_lumis,
                                               ## this is a dataset from the last pre-release: to be updated much too often IMO
                                               dataSet='/RelValTTbar_14TeV/CMSSW_12_6_0_pre4-PU_125X_mcRun3_2022_realistic_v4-v1/MINIAODSIM')}
steps['NANO_mc12.6']=merge([{'--era':'Run3',
                             '--conditions':'auto:phase1_2022_realistic'},
                            _NANO_mc])


################
#10.6 input
workflows[2500.31 ] = ['NANOmc106Xul16v2', ['TTbarMINIAOD10.6_UL16v2','NANO_mc10.6ul16v2', 'HRV_NANO_mc']]
workflows[2500.311] = ['NANOmc106Xul17v2', ['TTbarMINIAOD10.6_UL17v2','NANO_mc10.6ul17v2', 'HRV_NANO_mc']]
workflows[2500.312] = ['NANOmc106Xul18v2', ['TTbarMINIAOD10.6_UL18v2','NANO_mc10.6ul18v2', 'HRV_NANO_mc']]

workflows[2500.33 ] = ['NANOdata106Xul16v2', ['MuonEG2016MINIAOD10.6v2', 'NANO_data10.6ul16v2', 'HRV_NANO_data']]
workflows[2500.331] = ['NANOdata106Xul17v2', ['MuonEG2017MINIAOD10.6v2', 'NANO_data10.6ul17v2', 'HRV_NANO_data']]
workflows[2500.332] = ['NANOdata106Xul18v2', ['MuonEG2018MINIAOD10.6v2', 'NANO_data10.6ul18v2', 'HRV_NANO_data']]

################
#12.2 input
workflows[2500.401] = ['NANOmc122Xrun3', ['TTbarMINIAOD12.2','NANO_mc12.2', 'HRV_NANO_mc']]

################
#12.4 input
workflows[2500.501] = ['NANOmc124Xrun3', ['TTbarMINIAOD12.4','NANO_mc12.4', 'HRV_NANO_mc']]
workflows[2500.511] = ['NANOdata124Xrun3', ['MuonEG2022MINIAOD12.4','NANO_data12.4', 'HRV_NANO_data']]

################
#12.6 workflows
## these two workflows should be creating a sample "from scratch" instead of using a pre-release sample as input
workflows[2500.601] = ['NANOmc126X', ['TTBarMINIAOD12.6','NANO_mc12.6', 'HRV_NANO_mc']]

################
