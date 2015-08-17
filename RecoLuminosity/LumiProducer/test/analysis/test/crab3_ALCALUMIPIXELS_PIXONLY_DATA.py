from WMCore.Configuration import Configuration
config = Configuration()


#name='Pt15to30'
config.section_("General")
config.General.requestName = 'PCC_AlCaLumiPixels_Run2015A_PIXONLY_LS'
config.General.workArea = 'taskManagement'

config.section_("JobType")
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'Run_Pix_LS.py'
config.JobType.allowUndistributedCMSSW = True


#config.JobType.inputFiles = ['dttf_config.db']

config.section_("Data")
config.Data.inputDataset = '/AlCaLumiPixels/Run2015A-LumiPixels-PromptReco-v1/ALCARECO'
config.Data.lumiMask = ''

config.Data.ignoreLocality = True
#useParent = True


config.Data.inputDBS = 'global'
config.Data.splitting = 'LumiBased'
#config.Data.splitting = 'EventAwareLumiBased'
config.Data.publication = False
config.Data.unitsPerJob = 50
#config.Data.totalUnits = -1
#config.Data.publishDbsUrl = 'test'
config.Data.publishDataName = 'PCC_AlCaLumiPixels_Run2015A_PIXONLY_LS'
config.Data.outLFNDirBase = '/store/group/comm_luminosity/PCC/ForLumiComputation'

config.section_("Site")
config.Site.storageSite = 'T2_CH_CERN'
config.Site.whitelist=['T2_FR_CCIN2P3','T2_IT_Pisa','T2_UK_London_IC','T2_HU_Budapest']
#config.Site.whitelist=['T2_FR_CCIN2P3']
