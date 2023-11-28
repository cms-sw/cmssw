#------------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------------
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
from Configuration.StandardSequences.Eras import eras

#------------------------------------------------------------------------------------
# Options
#------------------------------------------------------------------------------------
options = VarParsing.VarParsing()

options.register('skipEvents',
                 0, # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Number of events to skip")

options.register('processEvents',
                 -1, # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Number of events to process")

options.register('inputFiles',
                 "file:inputFile.root",
                 VarParsing.VarParsing.multiplicity.list,
                 VarParsing.VarParsing.varType.string,
                 "Input files")

options.register('outputFile',
                 "file:hcalnano.root", # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Output file")

options.register('nThreads', 
                4, 
                VarParsing.VarParsing.multiplicity.singleton, 
                VarParsing.VarParsing.varType.int)

options.register('compressionAlgorithm', 
                "ZLIB", 
                VarParsing.VarParsing.multiplicity.singleton, 
                VarParsing.VarParsing.varType.string)

options.register('compressionLevel', 
                5, 
                VarParsing.VarParsing.multiplicity.singleton, 
                VarParsing.VarParsing.varType.int)

options.register('reportEvery', 
                100, 
                VarParsing.VarParsing.multiplicity.singleton, 
                VarParsing.VarParsing.varType.int)

options.parseArguments()

print(" ")
print("Using options:")
print(" skipEvents    =", options.skipEvents)
print(" processEvents =", options.processEvents)
print(" inputFiles    =", options.inputFiles)
print(" outputFile    =", options.outputFile)
print(" nThreads.     =", options.nThreads)
print(" ")

#------------------------------------------------------------------------------------
# Declare the process and input variables
#------------------------------------------------------------------------------------
process = cms.Process('HCALNANO', eras.Run3)

#------------------------------------------------------------------------------------
# Get and parse the command line arguments
#------------------------------------------------------------------------------------
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.processEvents) )

process.source = cms.Source(
    "PoolSource",
    fileNames  = cms.untracked.vstring(options.inputFiles),
    skipEvents = cms.untracked.uint32(options.skipEvents)
    )

process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string(options.outputFile)
    )
process.options.numberOfThreads=cms.untracked.uint32(options.nThreads)
process.options.numberOfStreams=cms.untracked.uint32(0)
#------------------------------------------------------------------------------------
# import of standard configurations
#------------------------------------------------------------------------------------

# Reduce message log output
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(options.reportEvery)

process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_Data_cff')

#------------------------------------------------------------------------------------
# Specify Global Tag
#------------------------------------------------------------------------------------
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = '122X_dataRun3_HLT_v3'
print("GlobalTag = ", str(process.GlobalTag.globaltag).split("'")[1])
print(" ")

#------------------------------------------------------------------------------------
# HcalNano sequence definition
#------------------------------------------------------------------------------------
#from PhysicsTools.NanoAOD.common_cff import *
process.load("PhysicsTools.NanoAOD.nano_cff")
process.load("RecoLocalCalo/Configuration/hcalLocalReco_cff")
process.load("RecoLocalCalo/Configuration/hcalGlobalReco_cff")
process.load("DPGAnalysis.HcalNanoAOD.hcalRecHitTable_cff")
process.load("DPGAnalysis.HcalNanoAOD.hcalDetIdTable_cff")
process.load("DPGAnalysis.HcalNanoAOD.hcalDigiSortedTable_cff")

# This creates a sorted list of HcalDetIds for use by downstream HcalNano table producers
process.hcalNanoPrep = cms.Sequence(process.hcalDetIdTable)

process.hcalNanoTask = cms.Task(
    process.hcalDigis, 

    ## Do energy reconstruction
    process.hcalLocalRecoTask,
    process.hcalGlobalRecoTask,

    # Make digi tables
    process.hcalDigiSortedTable,

    # Make RecHit tables
    process.hbheRecHitTable,
    process.hfRecHitTable,
    process.hoRecHitTable,
)

process.preparation = cms.Path(process.hcalNanoPrep, process.hcalNanoTask)

process.NanoAODEDMEventContent.outputCommands = cms.untracked.vstring(
        'drop *',
        "keep nanoaodFlatTable_*Table_*_*",     # event data
        "keep edmTriggerResults_*_*_*",  # event data
        "keep String_*_genModel_*",  # generator model data
        "keep nanoaodMergeableCounterTable_*Table_*_*", # accumulated per/run or per/lumi data
        "keep nanoaodUniqueString_nanoMetadata_*_*",   # basic metadata
    )

process.out = cms.OutputModule("NanoAODOutputModule",
    fileName = cms.untracked.string(options.outputFile),
    outputCommands = process.NanoAODEDMEventContent.outputCommands,
    compressionLevel = cms.untracked.int32(options.compressionLevel),
    compressionAlgorithm = cms.untracked.string(options.compressionAlgorithm),

)
process.end = cms.EndPath(process.out)  
