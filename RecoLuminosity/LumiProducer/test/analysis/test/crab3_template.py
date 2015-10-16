from WMCore.Configuration import Configuration
config = Configuration()


#name='Pt15to30'
config.section_("General")
config.General.requestName = 'PCC_ZeroBias_RAW_miniVdM_252116'
config.General.workArea = 'taskManagement'

config.section_("JobType")
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'RunFromRAW_AlwaysTrue_Data_Event.py'
config.JobType.allowUndistributedCMSSW = True


#config.JobType.inputFiles = ['dttf_config.db']

config.section_("Data")
config.Data.inputDataset = '/ZeroBias/Run2015B-v1/RAW'
config.Data.lumiMask = 'json_252116_optScan.txt'

config.Data.ignoreLocality = True
#useParent = True


config.Data.inputDBS = 'global'
config.Data.splitting = 'LumiBased'
#config.Data.splitting = 'EventAwareLumiBased'
config.Data.publication = False
config.Data.unitsPerJob = 4
#config.Data.totalUnits = -1
#config.Data.publishDbsUrl = 'test'
config.Data.publishDataName = 'PCC_ZeroBias_RAW_miniVdM_252116'

config.section_("Site")
config.Site.storageSite = 'T2_CH_CERN'
config.Site.whitelist=['T2_FR_CCIN2P3','T2_IT_Pisa','T2_UK_London_IC','T2_HU_Budapest']
#config.Site.whitelist=['T2_FR_CCIN2P3']
