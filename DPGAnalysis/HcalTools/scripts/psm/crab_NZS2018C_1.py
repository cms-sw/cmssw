from WMCore.Configuration import Configuration
config = Configuration()
#
config.section_('General')
config.General.transferOutputs = True
config.General.transferLogs = True
#
config.section_('JobType')
config.JobType.psetName = 'PSM_Global_2018_cfg.py'
config.JobType.pluginName = 'Analysis'
config.JobType.numCores = 4
config.JobType.outputFiles = ['Global.root']
#
config.section_('Data')
#config.Data.userInputFiles=['/store/data/Run2017E/AlCaLumiPixels/ALCARECO/AlCaPCCRandom-PromptReco-v1/000/303/832/00000/E6B8ACA4-95A4-E711-9AA2-02163E014793.root']
#config.Data.inputDataset = '/ZeroBias/Run2018C-v1/RAW'
config.Data.inputDataset = '/HcalNZS/Run2018C-v1/RAW'
#   main                    /HcalNZS/Run2018C-v1/RAW
#                           /MinimumBias/Run2018C-v1/RAW
#                           /ZeroBias/Run2018C-v1/RAW
#config.Data.inputDBS = 'global'
#
config.Data.unitsPerJob = 50
#config.Data.totalUnits = -1
#
config.Data.splitting = 'LumiBased'
#config.Data.splitting = 'FileBased'
#
#Example: '12345,99900-99910'
#config.Data.runRange = '272012,272021,272022'
#config.Data.runRange = '316590-320386'
#config.Data.runRange = '316590,316615,316667,316702,316717'
#config.Data.runRange = '320002,320006,320010,320011,320012'
#Run2018C: 
#319337 319347 319449 319486 319528 319625 319678 319756 319853 319910 
#319950 319991 320011 320038 320064 
config.Data.runRange = '319337'
config.Data.ignoreLocality = True
#config.Data.ignoreLocality = False
config.Data.lumiMask = 'https://cms-service-dqm.web.cern.ch/cms-service-dqm/CAF/certification/Collisions18/13TeV/PromptReco/Cert_314472-325175_13TeV_PromptReco_Collisions18_JSON.txt'
config.Data.outLFNDirBase= '/store/user/zhokin/PSM/HcalNZS/2018/319337/'
#config.Data.outLFNDirBase= '/store/user/kodolova/HCALPHISYM/2018/LEGACY/RUNC'
#config.Data.publication = False
#config.Data.publishDbsUrl = 'test'
#config.Data.outputDatasetTag = 'zzzzzzhokinTESTs'
#
#useParent = True
#
config.section_('Site')
#config.Site.storageSite = 'T3_US_FNALLPC'
config.Site.storageSite = 'T2_CH_CERN'
config.Site.whitelist = ['T2_CH_CERN']

