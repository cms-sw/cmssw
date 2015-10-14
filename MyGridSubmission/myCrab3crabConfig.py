from CRABClient.UserUtilities import config
config = config()

config.General.requestName = 'dummyRequestName'
config.General.workArea = 'crab_projects_ME0Segment_Timing'
config.General.transferLogs = True

config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'HGCal_config_RECO.py'
# config.JobType.outputFiles = ['triggerTree.root']  # in case the output is not an EDM ROOT file

config.Data.inputDataset = 'dummyInputDataSet'
# config.Data.inputDBS = 'global'                    # centraly produced DATA
config.Data.inputDBS = 'phys03'                      # localy produced DATA
# config.Data.publication = False
config.Data.publication = True
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 10                         # i should have taken 1 here ...
# config.Data.outLFN = '/store/user/piet/ME0Segment_Time/'
config.Data.outLFNDirBase = '/store/user/piet/ME0Segment_Time/'
config.Data.publishDataName = 'dummyPublishDataName'

config.Site.storageSite = 'T2_IT_Bari'
# config.Site.whitelist = ['T2_US_Wisconsin']
