# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step1 -s DQM -n 1 --eventcontent DQM --conditions auto:com10 --filein /store/relval/CMSSW_7_1_2/MinimumBias/RECO/GR_R_71_V7_dvmc_RelVal_mb2012Cdvmc-v1/00000/00209DF4-3708-E411-9FA7-0025905A6126.root --data --no_exec --python_filename=test_step1_cfg.py
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process('DQM')

options = VarParsing.VarParsing('analysis')
options.register('globalTag',
                 "132X_mcRun3_2023_realistic_v2", # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.string, # string, int, or float
                 "input file name")
options.register('sequenceType',
                 "electrons",
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.string, # string, int, or float
                 "type of sequence to run")
options.parseArguments()

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 10
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('DQMOffline.Configuration.DQMOffline_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
)

# Input source
process.source = cms.Source("PoolSource",
  secondaryFileNames = cms.untracked.vstring(),
                            fileNames = cms.untracked.vstring(options.inputFiles)
)

process.options = cms.untracked.PSet(
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.19 $'),
    annotation = cms.untracked.string('step1 nevts:1'),
    name = cms.untracked.string('Applications')
)

# Output definition

process.DQMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.DQMEventContent.outputCommands,
    fileName = cms.untracked.string('step1_DQM_1.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('')
    )
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag,options.globalTag, '')

# Tracker Data MC validation suite
process.load('DQM.TrackingMonitorSource.TrackingDataMCValidation_Standalone_cff')

minbias_analysis_step = cms.Path(process.standaloneValidationMinbias)
zee_analysis_step = cms.Path(process.standaloneValidationElec)
zmm_analysis_step = cms.Path(process.standaloneValidationMuon)
ttbar_analysis_step = cms.Path(process.standaloneValidationTTbar)

if(options.sequenceType == "electrons"):
    process.analysis_step = zee_analysis_step
elif (options.sequenceType == "muons") :
    process.analysis_step = zmm_analysis_step
elif (options.sequenceType == "ttbar") :
    process.analysis_step = ttbar_analysis_step
elif (options.sequenceType == "minbias") :
    process.analysis_step = minbias_analysis_step
else :
    raise RuntimeError("Unrecognized sequenceType given option: %. Exiting" % options.sequenceType)

# Path and EndPath definitions
process.endjob_step = cms.EndPath(process.endOfProcess)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.analysis_step, process.endjob_step, process.DQMoutput_step)

###################################################################
# Set the process to run multi-threaded
###################################################################
process.options.numberOfThreads = 8
