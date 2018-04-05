from CRABClient.UserUtilities import config, getUsernameFromSiteDB
config = config()

config.General.requestName = 'nano6'
config.General.workArea = 'crab_projects'
config.General.transferOutputs = True
config.General.transferLogs = True

config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'nano_80x_cfg.py'
#config.JobType.outputFiles = ['lzma.root']

config.Data.inputDataset = '/TT_TuneCUETP8M2T4_13TeV-powheg-pythia8/RunIISummer16MiniAODv2-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/MINIAODSIM'
config.Data.inputDBS = 'global'
config.Data.splitting = 'EventAwareLumiBased'
config.Data.unitsPerJob = 2000
config.Data.totalUnits = 6000
config.Data.outLFNDirBase = '/store/user/%s/NanoTest/' % (getUsernameFromSiteDB())
config.Data.publication = True
config.Data.outputDatasetTag = 'NanoTest2'

config.Site.storageSite = 'T2_IT_Pisa'
