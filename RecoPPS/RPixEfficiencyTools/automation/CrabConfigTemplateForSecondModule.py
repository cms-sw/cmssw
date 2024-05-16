from CRABClient.UserUtilities import config
config = config()
InputDataset="/Charmonium/Run2018B-12Nov2019_UL2018-v1/AOD"
GeometryFile="Geometry.VeryForwardGeometry.geometryRPFromDD_2018_cfi"

config.General.transferOutputs = True
config.General.transferLogs = True

config.JobType.scriptExe = 'wrapper.sh'
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = '/afs/cern.ch/user/l/lkita/CMSSW_11_3_2/src/RecoPPS/RPixEfficiencyTools/python/ReferenceAnalysisDQMWorker_cfg.py'
config.JobType.outputFiles = ["tmp.root"]
config.JobType.inputFiles = ["/eos/user/l/lkita/Charmonium/efficiency_reference.root"]
config.JobType.pyCfgParams = ["sourceFileList=/afs/cern.ch/user/l/lkita/CMSSW_11_3_2/src/RecoPPS/RPixEfficiencyTools/InputFiles/test.dat", "outputFileName=tmp.root", "efficiencyFileName=efficiency_reference.root"]
config.JobType.priority = 40

config.Data.outLFNDirBase = '/store/user/lkita'
config.Data.inputDataset = InputDataset
config.Data.publication = False
config.Data.inputDBS = 'global'
config.Data.splitting = 'LumiBased' 
config.Data.unitsPerJob = 1000
config.Data.runRange = '317080'
config.Data.lumiMask = "/afs/cern.ch/user/l/lkita/CMSSW_11_3_2/src/RecoPPS/RPixEfficiencyTools/InputFiles/test_mask.json"

config.Site.storageSite = 'T3_CH_CERNBOX'