runNumber = 355261
dataset = "/AlCaPPS/Run2022B-v2/RAW"
from CRABClient.UserUtilities import config
config = config()

config.General.requestName = 'AlCaPPS_Calib'+str(runNumber)
config.General.transferOutputs = True
config.General.transferLogs = True

config.JobType.pluginName = 'analysis'
config.JobType.psetName = '../../../CalibPPS/TimingCalibration/test/DiamondCalibrationWorker_cfg.py'
config.JobType.inputFiles = ['conditions.py']
config.JobType.allowUndistributedCMSSW = True
config.JobType.maxMemoryMB = 2500

config.Data.inputDataset = dataset
config.Data.runRange = str(runNumber)

config.Data.inputDBS = 'global'
config.Data.splitting = 'Automatic'  
config.Data.lumiMask = 'allrunsSB-PPS-forCalib.json'  
#config.Data.splitting = 'LumiBased'                                                                                             
# config.Data.splitting = 'FileBased'
#config.Data.unitsPerJob = 20

# If you want, you can mask with a JSON here, instead of using config.Data.runRange
config.Data.inputDBS = 'global'
# config.Data.lumiMask = '/eos/project/c/ctpps/Operations/DataExternalConditions/2018/CMSgolden_2RPGood_anyarms_EraB1.json'

config.Data.outLFNDirBase = '/store/group/dpg_ctpps/comm_ctpps/Commissioning_2022/diamond'
config.Data.publication = True
config.Data.outputDatasetTag = 'AlCaPPS_Calib'+str(runNumber)

config.Site.storageSite = 'T2_CH_CERN' 

# config.Site.blacklist = ['T1_US_FNAL']
