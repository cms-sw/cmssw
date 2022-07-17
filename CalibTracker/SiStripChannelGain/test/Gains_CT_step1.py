# Auto generated configuration file
# using: 
# Revision: 1.381.2.28 
# Source: /local/reps/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: step3 --datatier ALCARECO --conditions auto:com10 -s ALCA:PromptCalibProdSiStripGains --eventcontent ALCARECO -n 100 --dasquery=file dataset=/MinimumBias/Run2012C-SiStripCalMinBias-v2/ALCARECO run=200190 --fileout file:step3.root --no_exec
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing("analysis")
options.register("calibTreeName", "gainCalibrationTreeStdBunch/tree", VarParsing.multiplicity.singleton, VarParsing.varType.string, "Name of the TTree in the calibtree files")
options.register("firstRun", 0, VarParsing.multiplicity.singleton, VarParsing.varType.int, "First run of the calibration range")
options.register("lastRun", 0, VarParsing.multiplicity.singleton, VarParsing.varType.int, "Last run of the calibration range")
options.register("globalTag", "auto:run3_data_express", VarParsing.multiplicity.singleton, VarParsing.varType.string, "Global tag (express, to check the homogeneity of the calibration range")
options.parseArguments()

process = cms.Process('ALCA')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
process.SiStripCalib = DQMEDAnalyzer(
    "SiStripGainsCalibTreeWorker",
    minTrackMomentum    = cms.untracked.double(2),
    maxNrStrips         = cms.untracked.uint32(8),
    Validation          = cms.untracked.bool(False),
    OldGainRemoving     = cms.untracked.bool(False),
    FirstSetOfConstants = cms.untracked.bool(True),
    UseCalibration      = cms.untracked.bool(False),
    DQMdir              = cms.untracked.string('AlCaReco/SiStripGains'),
    calibrationMode     = cms.untracked.string('StdBunch'),
    ChargeHisto         = cms.untracked.vstring('TIB','TIB_layer_1','TOB','TOB_layer_1','TIDminus','TIDplus','TECminus','TECplus','TEC_thin','TEC_thick'),
    gain                = cms.untracked.PSet(
                            prefix = cms.untracked.string("GainCalibration"), 
                            suffix = cms.untracked.string('')
                            ),
    evtinfo             = cms.untracked.PSet(
                            prefix = cms.untracked.string(""), 
                            suffix = cms.untracked.string('')
                            ),
    tracks              = cms.untracked.PSet(
                            prefix = cms.untracked.string("track"), 
                            suffix = cms.untracked.string('')
                            ),
    CalibTreeName = cms.untracked.string(options.calibTreeName),
    )

process.SiStripCalib.CalibTreeFiles = cms.untracked.vstring(options.inputFiles)

process.MEtoEDMConvertSiStripGains = cms.EDProducer("MEtoEDMConverter",
        Name = cms.untracked.string('MEtoEDMConverter'),
        Verbosity = cms.untracked.int32(0), # 0 provides no output
        # 1 provides basic output
        # 2 provide more detailed output
        Frequency = cms.untracked.int32(50),
        MEPathToSave = cms.untracked.string('AlCaReco/SiStripGains'),
        deleteAfterCopy = cms.untracked.bool(True)
        )
process.seqALCARECOPromptCalibProdSiStripGains = cms.Sequence(
        process.SiStripCalib *
        process.MEtoEDMConvertSiStripGains
        )
process.pathALCARECOPromptCalibProdSiStripGains = cms.Path(process.seqALCARECOPromptCalibProdSiStripGains)

process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.Geometry.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("EmptyIOVSource",
   timetype   = cms.string('runnumber'),
   interval   = cms.uint64(1),
   firstValue = cms.uint64(options.firstRun),
   lastValue  = cms.uint64(options.lastRun)
)

# Uncomment to turn on verbosity output
process.load("FWCore.MessageLogger.MessageLogger_cfi")
#process.MessageLogger.threshold = cms.untracked.string('INFO')
process.MessageLogger.cout.threshold = cms.untracked.string('INFO')
process.MessageLogger.debugModules = cms.untracked.vstring("*")
#process.MessageLogger.destinations = cms.untracked.vstring('cout')
process.MessageLogger.cout = cms.untracked.PSet( threshold = cms.untracked.string('INFO'))

#process.Tracer = cms.Service("Tracer")

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    fileMode = cms.untracked.string('FULLMERGE')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.381.2.28 $'),
    annotation = cms.untracked.string('step3 nevts:100'),
    name = cms.untracked.string('PyReleaseValidation')
)

# Output definition

process.ALCARECOStreamPromptCalibProdSiStripGains = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('pathALCARECOPromptCalibProdSiStripGains')),
    outputCommands = cms.untracked.vstring(
        'keep *_alcaBeamSpotProducer_*_*',
        'keep *_MEtoEDMConvertSiStripGains_*_*'),
    fileName = cms.untracked.string(options.outputFile),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string("PromptCalibProdSiStripGains"),
        dataTier = cms.untracked.string('ALCARECO')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880)
)

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')

# Path and EndPath definitions
process.endjob_step = cms.EndPath(process.endOfProcess)
process.ALCARECOStreamPromptCalibProdSiStripGainsOutPath = cms.EndPath(process.ALCARECOStreamPromptCalibProdSiStripGains)

# Schedule definition
process.schedule = cms.Schedule(process.pathALCARECOPromptCalibProdSiStripGains,
                                process.endjob_step,
                                process.ALCARECOStreamPromptCalibProdSiStripGainsOutPath)
