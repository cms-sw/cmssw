### this is an example for running on RECO
### the name must be changed crab.cfg for actual running

from CRABClient.UserUtilities import config, getUsernameFromSiteDB
config = config()
outputName = 'Phase1Tracking_HI_development_v1'
config.General.requestName = outputName
config.General.workArea = outputName
config.General.transferOutputs = True
config.General.transferLogs = True

config.JobType.allowUndistributedCMSSW = True
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'step3_hi.py'
config.Data.inputDBS = 'phys03'
config.Data.inputDataset = '/Baty_2017Phase1TrackingGeom_Hydjet_GEN/abaty-Baty_2017Phase1TrackingGeom_Hydjet_RAW-20426d10a5dd9ea0ba2788dd87a0d614/USER'

config.Data.splitting = 'FileBased'
config.Data.ignoreLocality = False
config.Data.unitsPerJob = 1
config.Data.outLFNDirBase = '/store/user/%s/' % (getUsernameFromSiteDB())
config.Data.publication = False
config.Site.storageSite = 'T2_US_MIT'
