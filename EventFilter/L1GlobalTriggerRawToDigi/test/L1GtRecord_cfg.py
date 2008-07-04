#
# cfg file to produce L1GlobalTriggerRecord 
#


import FWCore.ParameterSet.Config as cms

# process
process = cms.Process("TestGtRecord")

# number of events to be processed and source file
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/g/ghete/scratch0/CmsswTestFiles/testGt_GtRecord_source.root')
)


# load and configure modules

# L1 EventSetup
process.load("L1Trigger.Configuration.L1DummyConfig_cff")

# L1 GT record producer
process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtRecord_cfi")

# input tag for GT readout collection: 
#     gtDigis       = GT emulator or GT unpacker (default)
#replace l1GtRecord.L1GtReadoutRecordTag = l1GtEmulDigis

# path to be run
process.p = cms.Path(process.l1GtRecord)

# services

# Message Logger
# uncomment / comment messages with DEBUG mode to run in DEBUG mode
process.MessageLogger = cms.Service("MessageLogger",
    testGt_GtRecord = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),      ## DEBUG mode 

        DEBUG = cms.untracked.PSet( 

            limit = cms.untracked.int32(-1)             ## DEBUG mode, all messages  
        ),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    debugModules = cms.untracked.vstring('l1GtRecord'), ## DEBUG mode 
    destinations = cms.untracked.vstring('testGt_GtRecord')
)

# output 
process.outputL1GtRecord = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testGt_GtRecord_output.root'),
    # keep only unpacked data in the ROOT file
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_l1GtRecord_*_*')
)

process.outpath = cms.EndPath(process.outputL1GtRecord)

