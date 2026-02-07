import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
from DQM.L1ScoutingMonitor.L1ScoutingMonitor_cfi import L1ScoutingMonitor

# Define command-line options
options = VarParsing.VarParsing('analysis')
options.register('nThreads', 4, options.multiplicity.singleton, options.varType.int, 'number of threads')
options.register('nStreams', 4, options.multiplicity.singleton, options.varType.int, 'number of streams')
options.parseArguments()

# Initialize process
process = cms.Process("scDQM")

# Load necessary modules
process.load("DQMServices.Core.DQMStore_cfi")

# Set options for number of threads and streams
process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(options.nThreads),
    numberOfStreams = cms.untracked.uint32(options.nStreams),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1),
)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100000

# Maximum number of events to process
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(-1))

# Source definition (example file in the same folder as the script)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        #'root://cms-xrd-global.cern.ch//store/data/Run2024D/L1Scouting/L1SCOUT/v1/000/380/346/00000/9a388b3c-78c4-46e7-a175-5f74fac407d1.root'
        'root://cms-xrd-global.cern.ch//store/data/Run2024G/L1Scouting/L1SCOUT/v1/000/385/415/00000/08043bc4-c19d-4c5c-a5a4-0fd44d8bf016.root'
    )
)

from DQM.L1ScoutingMonitor.L1ScoutingMonitor_cfi import L1ScoutingMonitor


# Load L1SMonitor configuration
process.L1ScoutingMonitor = L1ScoutingMonitor


process.DQMStore = cms.Service( "DQMStore",
    #enableMultiThread = cms.untracked.bool( True ),
    #trackME = cms.untracked.string( "" ),
    saveByLumi = cms.untracked.bool( True ),
    verbose = cms.untracked.int32( 0 )
)

# Example DQMFileSaverPB module
process.dqmFileSaverPB = cms.EDAnalyzer("DQMFileSaverPB",
    streamLabel = cms.untracked.string("streamDQMHistograms"),
    referenceHandling = cms.untracked.string("all"),
    producer = cms.untracked.string('DQM'),
    fakeFilterUnitMode = cms.untracked.bool(True), #True
    tag = cms.untracked.string('UNKNOWN'), #leave unknown, output name based on ls 
    referenceRequireStatus = cms.untracked.int32(100),
    path = cms.untracked.string("./")
)

process.dqmFileSaver = cms.EDAnalyzer( "DQMFileSaver",
    convention        = cms.untracked.string( "Online" ),
    workflow          = cms.untracked.string( "/L1Scouting/BX/Occupancy" ),
    dirName           = cms.untracked.string( "/" ),
    saveByRun         = cms.untracked.int32(1),
    saveByLumiSection = cms.untracked.int32(-1), 
    saveByEvent       = cms.untracked.int32(-1),
    saveByTime        = cms.untracked.int32(-1),
    saveByMinute      = cms.untracked.int32(-1),
    saveAtJobEnd      = cms.untracked.bool(False),
    forceRunNumber    = cms.untracked.int32(-1),
)

process.dqmFileSaver.saveByLumiSection=True




# End path configuration
process.endp = cms.EndPath(process.L1ScoutingMonitor  
                            + process.dqmFileSaverPB 
                            + process.dqmFileSaver
                            )


