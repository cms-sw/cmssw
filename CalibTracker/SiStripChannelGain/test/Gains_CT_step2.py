# Auto generated configuration file
# using: 
# Revision: 1.381.2.28 
# Source: /local/reps/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: step4 --data --conditions auto:com10 --scenario pp -s ALCAHARVEST:SiStripGains --filein file:PromptCalibProdSiStripGains.root -n -1 --no_exec
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing("analysis")
options.register("globalTag", "auto:run3_data_express", VarParsing.multiplicity.singleton, VarParsing.varType.string, "Global tag (express, to check the homogeneity of the calibration range)")
options.register("outputDbFile", "sqlite_file:promptCalibConditions.db", VarParsing.multiplicity.singleton, VarParsing.varType.string, "Connection string of output database")
options.register("fitMethod", "Legacy", VarParsing.multiplicity.singleton, VarParsing.varType.string, "Fit strategy (Legacy, DDRng, DDRngAllConv, or DDRngConvExceptTOBL5L6")
options.register("DQMOutput", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool, "Produce DQM output")
options.parseArguments()

process = cms.Process('ALCAHARVEST')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.AlCaHarvesting_cff')
process.load('Configuration.Geometry.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring(options.inputFiles),
    processingMode = cms.untracked.string('RunsAndLumis')
)


process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    fileMode = cms.untracked.string('FULLMERGE')
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.381.2.28 $'),
    annotation = cms.untracked.string('step4 nevts:-1'),
    name = cms.untracked.string('PyReleaseValidation')
)

# Output definition
process.load("Configuration.StandardSequences.DQMSaverAtJobEnd_cff") ## multi-run

## temporary workaround
process.load("FWCore.Services.InitRootHandlers_cfi")
process.InitRootHandlers.ResetRootErrHandler = cms.untracked.bool(False)

# Additional output definition

# Other statements
process.PoolDBOutputService.toPut.append(process.ALCAHARVESTSiStripGains_dbOutput)
process.pclMetadataWriter.recordsToMap.append(process.ALCAHARVESTSiStripGains_metadata)
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')

process.PoolDBOutputService.connect = cms.string(options.outputDbFile)

# Path and EndPath definitions
process.ALCAHARVESTDQMSaveAndMetadataWriter = cms.Path(process.dqmSaver+process.pclMetadataWriter)
process.SiStripGains = cms.Path(process.ALCAHARVESTSiStripGains)

process.dqmSaver.saveAtJobEnd = cms.untracked.bool(options.DQMOutput)

if options.outputFile:
    process.alcaSiStripGainsHarvester.StoreGainsTree = cms.untracked.bool(True)
    process.TFileService = cms.Service("TFileService", fileName = cms.string(options.outputFile))

process.alcaSiStripGainsHarvester.GoodFracForTagProd = cms.untracked.double(.95)
process.alcaSiStripGainsHarvester.NClustersForTagProd = cms.untracked.double(2.e8)
if options.fitMethod == "Legacy":
    process.alcaSiStripGainsHarvester.FitDataDrivenRange = cms.untracked.bool(False)
    process.alcaSiStripGainsHarvester.FitGaussianConvolution = cms.untracked.bool(False)
    process.alcaSiStripGainsHarvester.FitGaussianConvolutionTOBL5L6 = cms.untracked.bool(False)
elif options.fitMethod == "DDRng":
    process.alcaSiStripGainsHarvester.FitDataDrivenRange = cms.untracked.bool(True)
    process.alcaSiStripGainsHarvester.FitGaussianConvolution = cms.untracked.bool(False)
    process.alcaSiStripGainsHarvester.FitGaussianConvolutionTOBL5L6 = cms.untracked.bool(False)
elif options.fitMethod == "DDRngAllConv":
    process.alcaSiStripGainsHarvester.FitDataDrivenRange = cms.untracked.bool(True)
    process.alcaSiStripGainsHarvester.FitGaussianConvolution = cms.untracked.bool(True)
    process.alcaSiStripGainsHarvester.FitGaussianConvolutionTOBL5L6 = cms.untracked.bool(True)
elif options.fitMethod == "DDRngConvExceptTOBL5L6":
    process.alcaSiStripGainsHarvester.FitDataDrivenRange = cms.untracked.bool(True)
    process.alcaSiStripGainsHarvester.FitGaussianConvolution = cms.untracked.bool(True)
    process.alcaSiStripGainsHarvester.FitGaussianConvolutionTOBL5L6 = cms.untracked.bool(False)
else:
    raise RuntimeError("Unknown fit method: {0}".format(options.fitMethod))

# Schedule definition
process.schedule = cms.Schedule(process.SiStripGains,
                                process.ALCAHARVESTDQMSaveAndMetadataWriter)

#process.alcaSiStripGainsHarvester.calibrationMode = cms.untracked.string("IsoBunch")
