from CRABClient.UserUtilities import config, getUsernameFromSiteDB
config = config()

config.General.requestName = 'apvphaseproducertest_unstable_245194-245204_noCERN_v27'
config.General.workArea = '/afs/cern.ch/work/v/venturia/crab'
config.General.transferOutputs = True
config.General.transferLogs = True

config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'DPGAnalysis/SiStripTools/test/apvphaseproducertest_cfg.py'
config.JobType.pyCfgParams = ['globalTag=GR_P_V54']

config.Data.inputDataset = '/MinimumBias/Commissioning2015-PromptReco-v1/RECO'
config.Data.inputDBS = 'global'
config.Data.splitting = 'LumiBased'
config.Data.unitsPerJob = 100
#config.Data.totalUnits = 20
#config.Data.lumiMask = 'https://cms-service-dqm.web.cern.ch/cms-service-dqm/CAF/certification/Collisions12/8TeV/Reprocessing/Cert_190456-208686_8TeV_22Jan2013ReReco_Collisions12_JSON.txt'
config.Data.runRange = '245194,245195,245200,245204' # '193093-194075'
#config.Data.outLFNDirBase = '/store/user/%s/' % (getUsernameFromSiteDB())
config.Data.publication = False
#config.Data.publishDataName = 'CRAB3_tutorial_May2015_Data_analysis'

config.Site.storageSite = 'T2_IT_Pisa'
config.Site.blacklist = ['T2_CH_CERN']
