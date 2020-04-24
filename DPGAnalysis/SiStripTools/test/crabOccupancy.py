from CRABClient.UserUtilities import config
config = config()

config.General.requestName = 'Analysis'
config.General.workArea = 'CosmicsOccupancy2016B_v1REAL'
config.General.transferOutputs = True
config.General.transferLogs = True

config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'OccupancyPlotsTest_cfg.py'
config.JobType.pyCfgParams = ['globalTag=80X_dataRun2_Express_v6', 'tag=out', 'fromRAW=1', 'onCosmics=1']

#config.Data.primaryDataset = 'MinimumBias'
config.Data.inputDataset ='/Cosmics/Run2016B-v1/RAW'
#config.Data.inputDBS = 'phys03'
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 10
config.Data.ignoreLocality = True
#config.Data.totalUnits = 1 #500000
config.Data.outLFNDirBase = '/store/user/fiori/' # or '/store/group/<subdir>'
config.Data.publication = False
#config.Data.publishDataName = 'Bc2S1_AODSIM_RAW_8TeV_V1'
config.Site.storageSite ='T2_IT_Pisa'
#config.Site.whitelist =['T2_IT_Pisa']  #'T2_IT_Pisa' #'T3_TW_NTU_HEP'
