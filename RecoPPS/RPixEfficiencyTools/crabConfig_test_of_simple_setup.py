import CRABClient
from CRABClient.UserUtilities import config
config = config()

InputDataset ="/EGamma/Run2018B-12Nov2019_UL2018-v2/AOD"

config.General.transferOutputs = True
config.General.transferLogs = True

config.General.requestName = 'mobrzut_test_EA_DQM_Worker_3'
config.General.workArea = '/afs/cern.ch/user/m/mobrzut/automation/CMSSW_12_4_0/src/RecoPPS/RPixEfficiencyTools'



config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'python/EfficiencyAnalysisDQMWorker_cfg.py'
config.JobType.pyCfgParams = ["sourceFileList=/afs/cern.ch/user/m/mobrzut/public/Era.dat", "outputFileName=tmp.root"]
config.Data.inputDataset = InputDataset

config.Data.inputDBS = 'global'
config.Data.splitting = 'LumiBased'
# config.Data.splitting = 'Automatic'                                                                                         

config.Data.unitsPerJob = 20
config.Data.publication = False
config.Data.outLFNDirBase = '/store/group/dpg_ctpps/comm_ctpps/2018_PixelEfficiency'
config.Data.outputDatasetTag = 'CRAB3_tmobrzut_test_EA_DQM_Worker_3'
config.Data.runRange = '317080-317090'


config.Site.storageSite = 'T2_CH_CERN'
