from CRABClient.UserUtilities import config, getUsernameFromSiteDB
config = config()

config.General.requestName = 'apvphaseproducertest_minbias_relval_run2012D_v24'
config.General.workArea = '/afs/cern.ch/work/v/venturia/crab'
config.General.transferOutputs = True
config.General.transferLogs = True

config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'DPGAnalysis/SiStripTools/test/apvphaseproducertest_cfg.py'
config.JobType.pyCfgParams = ['globalTag=GR_R_75_V0A']

config.Data.inputDataset = '/MinimumBias/CMSSW_7_5_0_pre4-GR_R_75_V0A_RelVal_mb2012D-v1/RECO'
config.Data.inputDBS = 'global'
config.Data.splitting = 'LumiBased'
config.Data.unitsPerJob = 10
config.Data.totalUnits = 100
#config.Data.lumiMask = 'https://cms-service-dqm.web.cern.ch/cms-service-dqm/CAF/certification/Collisions12/8TeV/Prompt/Cert_190456-208686_8TeV_PromptReco_Collisions12_JSON.txt'
#config.Data.runRange = '193093-193999' # '193093-194075'
#config.Data.outLFNDirBase = '/store/user/%s/' % (getUsernameFromSiteDB())
config.Data.publication = False
#config.Data.publishDataName = 'CRAB3_tutorial_May2015_Data_analysis'

config.Site.storageSite = 'T2_IT_Pisa'
