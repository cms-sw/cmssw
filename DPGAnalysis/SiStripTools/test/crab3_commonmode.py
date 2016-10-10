from CRABClient.UserUtilities import config, getUsernameFromSiteDB
config = config()

#config.General.requestName = 'commonmode_zerobias_278509_278770_278808_v3'
config.General.requestName = 'commonmode_zerobias_277194_v3'
config.General.workArea = '/afs/cern.ch/work/v/venturia/crab'
config.General.transferOutputs = True
config.General.transferLogs = True

config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'DPGAnalysis/SiStripTools/test/commonmodeanalyzer_cfg.py'
config.JobType.pyCfgParams = ['globalTag=80X_dataRun2_Express_v9']

config.Data.inputDataset = '/ZeroBias/Run2016E-v2/RAW'
#config.Data.inputDataset = '/ZeroBias/Run2016F-v1/RAW'
config.Data.inputDBS = 'global'
config.Data.splitting = 'LumiBased'
config.Data.unitsPerJob = 40
#config.Data.totalUnits = 20
config.Data.lumiMask = "https://cms-service-dqm.web.cern.ch/cms-service-dqm/CAF/certification/Collisions16/13TeV/Cert_271036-278808_13TeV_PromptReco_Collisions16_JSON_NoL1T.txt"
#config.Data.runRange = '278770,278509,278808' # '193093-194075'
config.Data.runRange = '277194' # '193093-194075'
#config.Data.outLFNDirBase = '/store/user/%s/' % (getUsernameFromSiteDB())
config.Data.publication = False
#config.Data.publishDataName = 'CRAB3_tutorial_May2015_Data_analysis'

config.Site.storageSite = 'T2_IT_Pisa'

