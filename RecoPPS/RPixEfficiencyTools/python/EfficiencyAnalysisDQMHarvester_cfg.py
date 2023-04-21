import os

import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
import FWCore.ParameterSet.VarParsing as VarParsing
from Configuration.StandardSequences.Eras import eras
from Configuration.AlCa.GlobalTag import GlobalTag
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester


#SETUP PROCESS
process = cms.Process("EfficiencyAnalysisDQMHarvester", eras.Run3)


#SPECIFY INPUT PARAMETERS
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False),
    FailPath = cms.untracked.vstring('Type Mismatch')
    )
options = VarParsing.VarParsing()
options.register('inputFileNames',
                'outputEfficiencyAnalysisDQMWorker.root',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "input ROOT file names, comma-separated (file created by EfficiencyAnalysisDQMWorker)")

options.register('outputDirectoryPath',
                '.',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "directory in which the output ROOT file will be saved")

options.register('campaign',
                'testCampaign',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "")

options.register('workflow',
                'testWorkflow',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "")

options.register('dataPeriod',
                'testDataPeriod',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "")


options.parseArguments()


#PREPARE LOGGER
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    cout = cms.untracked.PSet( 
        optionalPSet = cms.untracked.bool(True),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noTimeStamps = cms.untracked.bool(False),
        FwkReport = cms.untracked.PSet(
            optionalPSet = cms.untracked.bool(True),
            reportEvery = cms.untracked.int32(10000),
            # reportEvery = cms.untracked.int32(1),
            limit = cms.untracked.int32(50000000)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        threshold = cms.untracked.string('INFO')
        ),
    categories = cms.untracked.vstring(
        "FwkReport"
        ),
)
process.MessageLogger.statistics = cms.untracked.vstring()


#LOAD NECCESSARY DEPENDENCIES
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("DQM.Integration.config.environment_cfi")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("Geometry.VeryForwardGeometry.geometryRPFromDD_2018_cfi")

#SETUP GLOBAL TAG
gt_from_env = os.getenv('EFFICIENCY_GT')
gt = gt_from_env
if gt == None:
    gt = 'auto:run3_data_prompt'

print('Using GT:',gt)
process.GlobalTag = GlobalTag(process.GlobalTag, gt)


#PREPARE SOURCE
file_names_list = options.inputFileNames.split(",")
input_files = ""
for file_name in file_names_list:
    input_files+="file:"+file_name+"\n"

print('Input files:\n',input_files,sep='')
process.source = cms.Source("DQMRootSource",
    fileNames = cms.untracked.vstring(input_files),
)

#SETUP HARVESTER
process.harvester = DQMEDHarvester('EfficiencyTool_2018DQMHarvester')


#CONFIGURE DQM Saver
process.dqmEnv.subSystemFolder = "RolCalPPS"
process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow = "/"+"/".join([options.campaign, options.workflow, options.dataPeriod])
process.dqmSaver.saveByRun = -1
process.dqmSaver.saveAtJobEnd = True
process.dqmSaver.forceRunNumber = 999999

print('Saving output in directory:',options.outputDirectoryPath)
process.dqmSaver.dirName = options.outputDirectoryPath

#SCHEDULE JOB
process.path = cms.Path(
      process.harvester
)

process.end_path = cms.EndPath(
    process.dqmSaver
)

process.schedule = cms.Schedule(
    process.path,
    process.end_path
)