#########################
#Author: Sam Higginbotham
########################
from WMCore.Configuration import Configuration
config = Configuration()


#name='Pt11to30'
config.section_("General")
config.General.requestName = 'PCC_Run2017E_Corrections'
config.General.workArea = 'RawPCCZeroBias2017'

config.section_("JobType")
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'raw_corr_Random_cfg.py'
config.JobType.allowUndistributedCMSSW = True
config.JobType.outputFiles = ['rawPCC.csv']

config.JobType.inputFiles = ['c.db']

config.section_("Data")
#config.Data.inputDataset = '/AlCaLumiPixels/Run2017E-AlCaPCCZeroBias-PromptReco-v1/ALCARECO'
config.Data.userInputFiles=['/store/data/Run2017E/AlCaLumiPixels/ALCARECO/AlCaPCCRandom-PromptReco-v1/000/303/832/00000/E6B8ACA4-95A4-E711-9AA2-02163E014793.root']
#config.Data.lumiMask = ''
#config.Data.runRange='303382'#,297283,297278,297280,297281,297271,297227,297230,297276,297261,297266'
config.Data.ignoreLocality = True
#useParent = True


config.Data.inputDBS = 'global'
#config.Data.splitting = 'LumiBased'
config.Data.splitting = 'FileBased'
config.Data.publication = False
config.Data.unitsPerJob = 1000
#config.Data.totalUnits = -1
#config.Data.publishDbsUrl = 'test'
config.Data.outputDatasetTag = 'PCC_AlCaLumiPixels_Run2017C_1kLS_NoZeroes'
config.Data.outLFNDirBase = '/store/group/comm_luminosity/PCC/ForLumiComputations/2017/5Feb2018'

config.section_("Site")
config.Site.storageSite = 'T2_CH_CERN'
config.Site.whitelist=['T2_FR_CCIN2P3','T2_IT_Pisa','T2_UK_London_IC','T2_HU_Budapest']
#config.Site.whitelist=['T2_FR_CCIN2P3']
