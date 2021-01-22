from __future__ import print_function
# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: SingleElectronPt10_cfi.py -s GEN,SIM,DIGI,L1 --pileup=NoPileUp --geometry DB --conditions=auto:startup -n 1 --no_exec
import FWCore.ParameterSet.Config as cms
import FWCore.Utilities.FileUtils as FileUtils # ADDED

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
                 "rxFiles",
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Output directory for buffer files")
options.register('nPayloadFrames',
                 13,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "N payload frames per event")
options.register('nHeaderFrames',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "N header frames per event")
options.register('nClearFrames',
                 0,
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

options.parseArguments()

if (options.maxEvents == -1):
    options.maxEvents = 1

fileList = FileUtils.loadListFromFile('ttbar.list')
readFiles = cms.untracked.vstring(*fileList)

# make output directory if it doesn't already exist
if not os.path.isdir(options.outDir): os.mkdir(options.outDir)

process = cms.Process('L1Emulator')

# import of standard configurations
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.Geometry.GeometryDB_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = readFiles,
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
    process.MessageLogger.debugModules = cms.untracked.vstring('*')
    process.MessageLogger.cerr.threshold = cms.untracked.string('DEBUG')


# TTree output file
process.load("CommonTools.UtilAlgos.TFileService_cfi")
process.TFileService.fileName = cms.string('l1tCalo_2019_histos.root')

process.load('L1Trigger.L1CaloTrigger.l1tS2PFJetInputPatternWriter_cfi')
process.l1tS2PFJetInputPatternWriter.pfTag = cms.InputTag("l1pfCandidates", "Puppi", "IN")#"REPR")
process.l1tS2PFJetInputPatternWriter.filename = cms.untracked.string(options.filename)
process.l1tS2PFJetInputPatternWriter.outDir = cms.untracked.string(options.outDir)
process.l1tS2PFJetInputPatternWriter.nPayloadFrames = cms.untracked.uint32(options.nPayloadFrames)
process.l1tS2PFJetInputPatternWriter.nHeaderFrames = cms.untracked.uint32(options.nHeaderFrames)
process.l1tS2PFJetInputPatternWriter.nClearFrames = cms.untracked.uint32(options.nClearFrames)

# Path and EndPath definitions
process.path = cms.Path(
    process.l1tS2PFJetInputPatternWriter)

if (options.edm):
    process.output_step = cms.EndPath(process.output)

