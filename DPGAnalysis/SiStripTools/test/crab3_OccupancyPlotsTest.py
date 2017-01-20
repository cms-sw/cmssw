from CRABClient.UserUtilities import config, getUsernameFromSiteDB
config = config()

#config.General.requestName = 'OccupancyPlotsTest_2016H_isolatedbunches_v4'
#config.General.requestName = 'OccupancyPlotsTest_2016H_ZB_isolatedbunches_v4'
config.General.requestName = 'OccupancyPlotsTest_2016F_ZB_isolatedbunches_v4'
config.General.workArea = '/afs/cern.ch/work/v/venturia/crab'
config.General.transferOutputs = True
config.General.transferLogs = True

config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'DPGAnalysis/SiStripTools/test/OccupancyPlotsTest_cfg.py'
config.JobType.pyCfgParams = ['globalTag=80X_dataRun2_Prompt_v14','fromRAW=1','tag=2016F_ZB_isolatedbunches','triggerPath=HLT_ZeroBias_v*']
#config.JobType.pyCfgParams = ['globalTag=80X_dataRun2_Prompt_v14','fromRAW=1','tag=2016H_isolatedbunches']

#config.Data.inputDataset = '/MinimumBias/Run2012A-22Jan2013-v1/RECO'
#config.Data.inputDataset = '/ZeroBias/Run2016H-v1/RAW'
config.Data.inputDataset = '/ZeroBias/Run2016F-v1/RAW'
config.Data.inputDBS = 'global'
config.Data.splitting = 'LumiBased'
config.Data.unitsPerJob = 50
#config.Data.totalUnits = 20
#config.Data.lumiMask = 'https://cms-service-dqm.web.cern.ch/cms-service-dqm/CAF/certification/Collisions16/13TeV/Cert_271036-283059_13TeV_PromptReco_Collisions16_JSON_NoL1T.txt'
config.Data.lumiMask = 'JSON_2016_isolatedbunches.txt'
#config.Data.runRange = '278761,277932,277933,277934,277935,282649,282650,284077,284078' # '193093-194075'
#config.Data.outLFNDirBase = '/store/user/%s/' % (getUsernameFromSiteDB())
config.Data.publication = False
#config.Data.publishDataName = 'CRAB3_tutorial_May2015_Data_analysis'

config.Site.storageSite = 'T2_IT_Pisa'

