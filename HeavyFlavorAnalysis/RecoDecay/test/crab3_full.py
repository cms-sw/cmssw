from WMCore.Configuration import Configuration
config = Configuration()

config.section_('General')
config.General.requestName = 'BsToJPsiPhi_full'
# request name is the name of the folder where crab log is saved

config.General.workArea = 'crab3_projects'
config.General.transferOutputs = True

config.section_('JobType')
# set here the path of your working area
config.JobType.psetName = '.../src/HeavyFlavorAnalysis/RecoDecay/test/cfg_full.py'
config.JobType.pluginName = 'Analysis'
config.JobType.outputFiles = ['dump_full.txt','hist_full.root']
config.JobType.allowUndistributedCMSSW = True

config.section_('Data')

config.Data.inputDataset = '/BsToJpsiPhi_BMuonFilter_TuneCUEP8M1_13TeV-pythia8-evtgen/RunIISpring15DR74-Asympt25nsRaw_MCRUN2_74_V9-v1/AODSIM' 

config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 5

# set here the path of a storage area you can write to
config.Data.outLFNDirBase = '/store/user/...'
config.Data.publication = False

############## 

#config.Data.ignoreLocality = True

config.section_("Site")

# set here a storage site you can write to
config.Site.storageSite = 'T2_IT_Legnaro'





