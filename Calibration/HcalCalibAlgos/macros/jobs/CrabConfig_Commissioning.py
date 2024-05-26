from CRABClient.UserUtilities import config
config = config()

config.General.requestName = 'IsoTrk_2024C_Comissioning_v2'
config.General.transferLogs = True
config.General.transferOutputs = True
config.General.workArea = 'CRAB3_IsoTrkCalib_Comissioning_2024C_v2'

config.JobType.maxMemoryMB = 4000
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'isoTrackAlCaCommissioning_cfg.py'
#config.JobType.psetName = 'isoTrackAlCaCounter_cfg.py'
config.JobType.allowUndistributedCMSSW = True
#config.Data.userInputFiles = open('input_files_Mahi22C.txt').readlines()
config.Data.inputDataset = '/Commissioning/Run2024C-HcalCalIsoTrk-PromptReco-v1/ALCARECO'
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 1
#config.Data.totalUnits = 5
config.Data.publication = False
#config.Data.ignoreLocality = False
#config.Data.outputDatasetTag = 'collision13.6TeV'
#config.Data.runRange = '367080-367764'
config.Data.outLFNDirBase = '/store/group/dpg_hcal/comm_hcal/suman/13p6TeV/2024EraB/'
#config.Data.lumiMask ='https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions23/Cert_Collisions2023_eraC_367095_368823_Golden.json'

config.Site.storageSite = 'T2_CH_CERN'
#config.Site.whitelist = ['T2_CH_CERN']
