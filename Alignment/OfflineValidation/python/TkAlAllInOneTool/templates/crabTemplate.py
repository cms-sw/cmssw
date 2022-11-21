from WMCore.Configuration import Configuration
from CRABClient.UserUtilities import getUsername

config = Configuration()

inputList = 'inputFiles.txt'
jobTag = "exampleJobName"
username = getUsername()

config.section_("General")
config.General.requestName = jobTag
config.General.workArea = config.General.requestName
config.General.transferOutputs = True
config.General.transferLogs = False

config.section_("JobType")
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'validation_cfg.py'
config.JobType.pyCfgParams = ['config=validation.json', 'runType=crab']
config.JobType.inputFiles = ['validation.json']
config.JobType.numCores = 1
config.JobType.maxMemoryMB = 1200
config.JobType.maxJobRuntimeMin = 900

config.section_("Data")
config.Data.userInputFiles = open(inputList).readlines()
config.Data.totalUnits = len(config.Data.userInputFiles)
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 1
config.Data.outputPrimaryDataset = 'AlignmentValidation'
config.Data.outLFNDirBase = '/store/group/alca_trackeralign/' + username + '/' + config.General.requestName
config.Data.publication = False

config.section_("Site")
config.Site.whitelist = ['T2_CH_*','T2_DE_*','T2_FR_*','T2_IT_*']
config.Site.storageSite = 'T2_CH_CERN'
