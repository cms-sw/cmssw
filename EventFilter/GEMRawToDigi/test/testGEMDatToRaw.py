# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: SingleElectronPt10_cfi.py -s GEN,SIM,DIGI,L1 --pileup=NoPileUp --geometry DB --conditions=auto:startup -n 1 --no_exec
import FWCore.ParameterSet.Config as cms

process = cms.Process('RAW')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.Geometry.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1))

process.maxEvents.input = cms.untracked.int32(10)

process.options = cms.untracked.PSet(
    SkipEvent = cms.untracked.vstring('ProductNotFound')
)

# Output definition

process.output = cms.OutputModule(
    "PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = cms.untracked.vstring("keep *",
					   "drop *_mix_*_*"),
    fileName = cms.untracked.string('out.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('')
    )
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')

# TTree output file
process.load("CommonTools.UtilAlgos.TFileService_cfi")
process.TFileService.fileName = cms.string('l1t.root')

process.source = cms.Source("GEMDataInputSource",
    runNumber = cms.untracked.int32(-1),
    fileNames = cms.untracked.vstring('file:run304140_ls0001_streamA_StorageManager.dat'),
    fedid = cms.untracked.int32(1467)
)

# dump raw data
process.dumpRaw = cms.EDAnalyzer("DumpFEDRawDataProduct",
    label = cms.untracked.string("source"),
    feds = cms.untracked.vint32 ( 1467 ),
    dumpPayload = cms.untracked.bool ( True )
)

# raw to digi
process.load('EventFilter.GEMRawToDigi.gemRawToDigi_cfi')
process.load('EventFilter.GEMRawToDigi.GEMSQLiteCabling_cfi')
process.gemRawToDigi.InputObjects = cms.InputTag('source')

# enable debug message logging for our modules
process.MessageLogger = cms.Service("MessageLogger",
    threshold  = cms.untracked.string('INFO'),
    debugModules = cms.untracked.vstring('*'),
    categories = cms.untracked.vstring("GEMDataInputSource")
)

# Path and EndPath definitions
process.path = cms.Path(
    process.gemRawToDigi
    #process.dumpRaw+process.gemRawToDigi
)

process.out = cms.EndPath(
    process.output
)
