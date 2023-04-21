from CRABClient.UserUtilities import config
config = config()

config.General.requestName = 'ctpps_2018B1_pixelefficiency_EGamma'
config.General.workArea = '/afs/cern.ch/user/l/lkita/CMSSW_11_3_2/src/RecoPPS/RPixEfficiencyTools'
config.General.transferOutputs = True
config.General.transferLogs = True
# config.General.instance = 'preprod' 

config.JobType.pluginName = 'analysis'
config.JobType.psetName = 'test/skimFilesForCRAB.py'
config.JobType.allowUndistributedCMSSW = True

config.Data.inputDataset = "/EGamma/Run2018B-12Nov2019_UL2018-v2/AOD"
# '/EGamma/Run2018C-12Nov2019_UL2018-v2/AOD'
# '/EGamma/Run2018D-12Nov2019_UL2018-v4/AOD'
# config.Data.allowNonValidInputDataset = True # CAREFUL WITH THIS

config.Data.inputDBS = 'global'

config.Data.splitting = 'FileBased'      
# config.Data.splitting = 'Automatic'                                                                                         
config.Data.unitsPerJob = 20                                                                                                      
config.Data.lumiMask = '/eos/project/c/ctpps/Operations/DataExternalConditions/2018/CMSgolden_2RPGood_anyarms_EraB1.json'
# config.Data.totalUnits = 10

config.Data.outLFNDirBase = '/store/group/dpg_ctpps/comm_ctpps/2018_PixelEfficiency'
config.Data.publication = False
config.Data.outputDatasetTag = 'ctpps_2018B1_pixelefficiency_singleele'

config.Site.storageSite = 'T2_CH_CERN'
config.Site.blacklist = ['T3_UK_London_QMUL','T3_UK_London_RHUL','T2_PL_Swierk','T2_FR_GRIF_IRFU']