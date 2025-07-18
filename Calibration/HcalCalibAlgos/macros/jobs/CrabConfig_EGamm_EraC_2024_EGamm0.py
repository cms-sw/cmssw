import CRABClient
from CRABClient.UserUtilities import config
config = config()

config.General.requestName = 'EGamma0_2024_EraC_v1'
#config.General.transferOutputs = True
config.General.transferLogs = False
config.JobType.maxMemoryMB = 2500
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'isoTrackNewAlCaAnalysis_cfg.py' #isoTrackAlCaAnalysis_cfg.py'

#config.Data.inputDataset = '/MinimumBias/Run2017D-21Sep2017-v1/RECO'
#config.Data.inputDataset = '/MinimumBias0/Commissioning2021-HcalCalIsoTrkFilter-PromptReco-v1/ALCARECO'
config.Data.inputDataset = '/EGamma0/Run2024C-HcalCalIsoTrkProducerFilter-PromptReco-v1/ALCARECO'
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 5
#config.Data.totalUnits = -1
config.Data.publication = False
config.Data.outputDatasetTag = config.General.requestName
config.Data.partialDataset = True
#config.Data.runRange = '379415-379729'
#https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/Cert_Collisions2022_355100_356175_Muon.json'
#config.Data.lumiMask = 'https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions24/Cert_Collisions2024_378981_379075_Golden.json'
#https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions23/Cert_Collisions2023_eraB_366403_367079_Golden.json'
config.Site.storageSite = 'T2_CH_CERN'
config.Data.outLFNDirBase  = '/store/group/dpg_hcal/comm_hcal/Jyoti/13p6TeV/2024/EraC/EGamma0_v1/'
