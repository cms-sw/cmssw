from WMCore.Configuration import Configuration
config = Configuration()


#name='Pt15to30'
config.section_("General")
config.General.requestName = 'PCC_outputdirname'
config.General.workArea = 'taskManagement'

config.section_("JobType")
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'RunFromRAW_template.py'
config.JobType.allowUndistributedCMSSW = True


#config.JobType.inputFiles = ['dttf_config.db']

config.section_("Data")
config.Data.inputDataset = '/Neutrino_Pt-2to20_gun/Phys14DR-Flat20to50BX50_tsg_PHYS14_ST_V1-v1/GEN-SIM-RAW'


config.Data.ignoreLocality = True
#useParent = True


config.Data.inputDBS = 'global'
config.Data.splitting = 'LumiBased'
#config.Data.splitting = 'EventAwareLumiBased'
config.Data.publication = False
config.Data.unitsPerJob = 400
#config.Data.totalUnits = -1
#config.Data.publishDbsUrl = 'test'
config.Data.publishDataName = 'PCC_outputdirname'

config.section_("Site")
config.Site.storageSite = 'T2_CH_CERN'
config.Site.whitelist=['T2_FR_CCIN2P3','T2_IT_Pisa','T2_UK_London_IC','T2_HU_Budapest']
#config.Site.whitelist=['T2_FR_CCIN2P3']
