import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

process = cms.Process("reader")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.threshold = cms.untracked.string('INFO')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

options = VarParsing ('analysis')
options.register ("firstRun", 341761, VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.maxEvents = 10000
options.parseArguments()

# Better to know actual number of events in the .raw data file to set maxEvents.
# Otherwise it doesn't stop automatically at the end of reading of .raw data file
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) )

process.source = cms.Source("EmptySource",
      firstRun= cms.untracked.uint32(options.firstRun),
      numberEventsInLuminosityBlock = cms.untracked.uint32(200),
      numberEventsInRun       = cms.untracked.uint32(0)
)

# For B904 setup ME11 chamber, which corresponds to ME+1/1/02 in the production system mapping
#   changing to FED837 and RUI16 could let to pass data without disabling mapping consistency check unpacking flags
process.rawDataCollector = cms.EDProducer('CSCFileReader',
      firstEvent  = cms.untracked.int32(0),
      FED846 = cms.untracked.vstring('RUI01'),
      RUI01 = cms.untracked.vstring(options.inputFiles[0])
#      FED837 = cms.untracked.vstring('RUI16'),
#      RUI16 = cms.untracked.vstring('/afs/cern.ch/user/b/barvic/public/cscgem_tests/csc_00000001_EmuRUI01_Local_000_210519_162820_UTC.raw')
)

process.FEVT = cms.OutputModule("PoolOutputModule",
        fileName = cms.untracked.string(options.inputFiles[0].replace('.raw','.root')),
        outputCommands = cms.untracked.vstring("keep *")
)

process.p = cms.Path( process.rawDataCollector)

process.outpath = cms.EndPath(process.FEVT)
