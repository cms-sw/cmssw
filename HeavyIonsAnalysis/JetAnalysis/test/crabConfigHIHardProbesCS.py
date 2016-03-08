from CRABClient.UserUtilities import config, getUsernameFromSiteDB
config = config()

config.section_('General')
config.General.requestName = 'HIHardProbes_v4'
config.General.workArea = 'crab_projects'
config.General.transferOutputs = True
config.General.transferLogs = False


config.section_('JobType')
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'runForestAOD_PbPb_DATA_75X_CS.py'

config.section_('Data')
config.Data.inputDataset = '/HIHardProbes/HIRun2015-PromptReco-v1/AOD'
#config.Data.inputDBS = 'phys03'
config.Data.splitting = "EventAwareLumiBased"
config.Data.unitsPerJob = 10000
#config.Data.totalUnits = 100000
config.Data.lumiMask = '/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions15/HI/Cert_262548-263757_PromptReco_HICollisions15_JSON_v2.txt'
config.Data.outLFNDirBase = '/store/group/cmst3/group/hintt/mverweij/CS/data'
config.Data.publication = False #True
config.Data.outputDatasetTag = ''

config.section_('User')
config.section_('Site')
#config.Site.whitelist = ['T2_US_MIT']
#config.Site.blacklist = ['T2_US_Nebraska','T2_US_Florida','T2_US_Wisconsin','T2_US_Caltech']
config.Site.storageSite = 'T2_CH_CERN'

