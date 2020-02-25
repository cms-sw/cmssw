from CRABClient.UserUtilities import config, getUsernameFromSiteDB
config = config()

config.General.requestName = 'test18_crab_pps_toymc'
config.General.workArea = 'crab_test18_pps_toymc'
config.General.transferOutputs = True
config.General.transferLogs = False

config.Data.inputDBS = 'phys03'
config.JobType.pluginName = 'PrivateMC'
config.JobType.psetName = '950_postTS2_120.py'
config.JobType.allowUndistributedCMSSW = True

#config.JobType.inputFiles = ['2016_postTS2.xml', '2016_preTS2.xml', '2017_postTS2.xml', '2017_preTS2.xml', '2018.xml']
#config.JobType.inputFiles = ['/afs/cern.ch/user/d/dmf/private/work/private/CMSPhysicsAnalysis/PrivateMCProduction/2019_production_13TeV/PPS/PPS_MC_Production/CMSSW_10_6_8_patch1/src/Validation/CTPPS/alignment']
config.Data.outputPrimaryDataset = 'Test18PPSToyMC'
config.Data.splitting = 'EventBased'
config.Data.unitsPerJob = 100
NJOBS = 100  # This is not a configuration parameter, but an auxiliary variable that we use in the next line.
config.Data.totalUnits = config.Data.unitsPerJob * NJOBS
config.Data.outLFNDirBase = '/store/user/%s/' % (getUsernameFromSiteDB())
config.Data.publication = True
config.Data.outputDatasetTag = 'CRAB3_test18_PPS_TOY_MC'
config.Site.storageSite = 'T2_CH_CERN'
