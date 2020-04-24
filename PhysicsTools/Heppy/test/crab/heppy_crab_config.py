from WMCore.Configuration import Configuration
config = Configuration()

config.section_("General")
config.General.requestName = 'CRAB_HEPPY_test_2'
config.General.workArea = 'crab_projects_test_2'

config.section_("JobType")
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'heppy_crab_fake_pset.py'
config.JobType.scriptExe = 'heppy_crab_script.sh'
config.JobType.inputFiles = ['heppy_config.py','heppy_crab_script.py']
config.JobType.outputFiles = ['tree.root']

config.section_("Data")
config.Data.inputDataset = '/WH_HToBB_WToLNu_M-125_13TeV_powheg-herwigpp/Phys14DR-PU40bx25_PHYS14_25_V1-v1/MINIAODSIM'
config.Data.inputDBS = 'global'
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 1
config.Data.outLFN = '/store/user/arizzi/CRABHeppyTest1/'
config.Data.publication = True
config.Data.publishDataName = 'CRAB_HEPPY_Test1'

config.section_("Site")
config.Site.storageSite = "T2_IT_Rome"

#if you uncomment this you have to change also
#the heppy_crab_script.py uncommenting the line
#      #crabFiles[i]="root://cms-xrd-global.cern.ch/"+crabFiles[i]
#config.Data.ignoreLocality = True
