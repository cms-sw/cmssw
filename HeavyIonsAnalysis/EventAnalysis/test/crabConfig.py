from WMCore.Configuration import Configuration

config = Configuration()

config.section_("General")

config.General.requestName   = 'eventana_Hydjet_AOD3'
config.General.transferLogs = True

config.section_("JobType")
config.JobType.pluginName  = 'Analysis'
#config.JobType.psetName    = 'hiEvtAnalyzer_mc_cfg.py'
config.JobType.psetName    = 'hiEvtAnalyzer_data_cfg.py'

config.section_("Data")
#config.Data.inputDataset = '/Hydjet_Quenched_MinBias_5020GeV/tuos-HydjetMB_AOD_750pre5_round3v01-81dd8ce0064b5342f0d7e3ef953b6d47/USER'
config.Data.inputDataset = '/HIMinBiasUPC/tuos-MB_AOD_DATA750pre5_round3v01-f884d8d44fa079fef73ed78a04ab1ee3/USER'
config.Data.inputDBS = 'phys03'
config.Data.splitting = 'FileBased'
#config.Data.splitting = 'LumiBased'
config.Data.unitsPerJob = 1
config.Data.outLFNDirBase = '/store/group/phys_heavyions/azsigmon/PbPb2015/'

config.section_('Site')
config.Site.storageSite = 'T2_CH_CERN'
