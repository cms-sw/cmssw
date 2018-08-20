from __future__ import print_function
# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: SingleElectronPt10_cfi.py -s GEN,SIM,DIGI,L1 --pileup=NoPileUp --geometry DB --conditions=auto:startup -n 1 --no_exec
import FWCore.ParameterSet.Config as cms
import os

# options
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing('analysis')
options.register('filename',
                 "rx_summary",
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Filename string")
options.register('outDir',
                 "mpRxFiles",
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Output directory for buffer files")
options.register('mpPayloadFrames',
                 40,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "N payload frames per event")
options.register('mpHeaderFrames',
                 1,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "N header frames per event")
options.register('mpClearFrames',
                 13,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "N clear frames between events")
options.register('dump',
                 True,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Dump messages")
options.register('pattern',
                 True,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Save pattern file")
options.register('edm',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Save EDM file")
options.register('debug',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Enable debug data")
#options.register('doMP',
#                 True,
#                 VarParsing.VarParsing.multiplicity.singleton,
#                 VarParsing.VarParsing.varType.bool,
#                 "Generate MP pattern")
#options.register('doDemux',
#                 True,
#                 VarParsing.VarParsing.multiplicity.singleton,
#                 VarParsing.VarParsing.varType.bool,
#                 "Read demux data")
#options.register('nMP',
#                 1,
#                 VarParsing.VarParsing.multiplicity.singleton,
#                 VarParsing.VarParsing.varType.int,
#                 "Number of MPs")

options.parseArguments()

if (options.maxEvents == -1):
    options.maxEvents = 1

# make output directory if it doesn't already exist
try:
    os.stat(options.outDir)
except:
    print('Output directory does not exist. Creating directory: ' + options.outDir)
    os.mkdir(options.outDir)

process = cms.Process('L1Emulator')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.Geometry.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic8TeVCollision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

# import of standard configurations
process.load('Configuration.StandardSequences.RawToDigi_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
)

# Input source
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(options.inputFiles)
#        '/store/relval/CMSSW_7_4_0/RelValSingleElectronPt35_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/MCRUN2_74_V7_GENSIM_7_1_15-v1/00000/1E628CEB-7ADD-E411-ACF3-0025905A609E.root'
#        )
)

process.options = cms.untracked.PSet(

)

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')


# Output definition
process.output = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = cms.untracked.vstring("keep *",
					   "drop *_mix_*_*"),
    fileName = cms.untracked.string('SingleElectronPt10_cfi_py_GEN_SIM_DIGI_L1.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    )
)


# enable debug message logging for our modules
process.MessageLogger.categories.append('L1TCaloEvents')

process.MessageLogger.suppressInfo = cms.untracked.vstring('Geometry', 'AfterSource')

if (options.dump):
    process.MessageLogger.infos.placeholder = cms.untracked.bool(False)
    process.MessageLogger.infos.INFO = cms.untracked.PSet(limit = cms.untracked.int32(0))
    process.MessageLogger.infos.L1TCaloEvents = cms.untracked.PSet(
      optionalPSet = cms.untracked.bool(True),
      limit = cms.untracked.int32(10000)
    )

if (options.debug):
#    process.MessageLogger.debugModules = cms.untracked.vstring('L1TRawToDigi:caloStage2Digis', 'MP7BufferDumpToRaw:stage2MPRaw', 'MP7BufferDumpToRaw:stage2DemuxRaw')
    process.MessageLogger.debugModules = cms.untracked.vstring('*')
    process.MessageLogger.cerr.threshold = cms.untracked.string('DEBUG')


# TTree output file
process.load("CommonTools.UtilAlgos.TFileService_cfi")
process.TFileService.fileName = cms.string('l1tCalo_2016_histos.root')

# upgrade calo stage 2
process.load('L1Trigger.L1TCalorimeter.caloStage2Params_2017_v1_4_cfi')
process.caloStage2Params.towerEncoding    = cms.bool(True)
process.load('L1Trigger.L1TCalorimeter.simCaloStage2Layer1Digis_cfi')
process.load('L1Trigger.L1TCalorimeter.simCaloStage2Digis_cfi')
process.simCaloStage2Digis.useStaticConfig = True
process.simCaloStage2Layer1Digis.ecalToken = cms.InputTag("ecalDigis:EcalTriggerPrimitives")
process.simCaloStage2Layer1Digis.hcalToken = cms.InputTag("hcalDigis")

process.load('L1Trigger.L1TCalorimeter.l1tStage2InputPatternWriter_cfi')
process.l1tStage2InputPatternWriter.filename = cms.untracked.string(options.filename)
process.l1tStage2InputPatternWriter.outDir = cms.untracked.string(options.outDir)
process.l1tStage2InputPatternWriter.mpPayloadFrames = cms.untracked.uint32(options.mpPayloadFrames)
process.l1tStage2InputPatternWriter.mpHeaderFrames = cms.untracked.uint32(options.mpHeaderFrames)
process.l1tStage2InputPatternWriter.mpClearFrames = cms.untracked.uint32(options.mpClearFrames)

# Path and EndPath definitions
process.path = cms.Path(
    process.ecalDigis
    +process.hcalDigis
    +process.simCaloStage2Layer1Digis
    +process.simCaloStage2Digis
    +process.l1tStage2InputPatternWriter)

if (options.edm):
    process.output_step = cms.EndPath(process.output)

