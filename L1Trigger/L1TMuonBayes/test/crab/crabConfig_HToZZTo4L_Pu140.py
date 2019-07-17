from CRABClient.UserUtilities import config, getUsernameFromSiteDB
config = config()

config.General.requestName = 'omtf_MC_analysis_HToZZTo4L_Pu140'
#config.General.workArea = 'crab_projects'
config.General.transferOutputs = True
config.General.transferLogs = True

config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'runMuonOverlapTTMergerAnalyzerCrab.py'

config.Data.inputDataset = '/VBF_HToZZTo4L_M125_14TeV_powheg2_JHUgenV702_pythia8/PhaseIIFall17D-L1TPU140_93X_upgrade2023_realistic_v5-v1/GEN-SIM-DIGI-RAW'
#'/SingleNeutrino/PhaseIIFall17D-L1TPU200_93X_upgrade2023_realistic_v5-v1/GEN-SIM-DIGI-RAW'
#'/QCD_Pt-0to1000_Tune4C_14TeV_pythia8/PhaseIIFall17D-L1TPU200_93X_upgrade2023_realistic_v5-v1/GEN-SIM-DIGI-RAW'
#config.Data.inputDataset = '/SingleMu_FlatPt-2to100/PhaseIIFall17D-L1TPU200_93X_upgrade2023_realistic_v5-v1/GEN-SIM-DIGI-RAW'
#config.Data.inputDataset = '/SingleMu_FlatPt-2to100/PhaseIIFall17D-L1TnoPU_93X_upgrade2023_realistic_v5-v1/GEN-SIM-DIGI-RAW'

config.Data.inputDBS = 'global'
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 5
config.Data.outLFNDirBase = '/store/user/%s/' % (getUsernameFromSiteDB())
config.Data.publication = False
config.Data.outputDatasetTag = 'CRAB3_omtf_MC_analysis'
config.Data.totalUnits = 100

config.Site.storageSite = 'T2_PL_Swierk'
