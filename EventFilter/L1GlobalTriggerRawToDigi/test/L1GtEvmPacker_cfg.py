#
# cfg file to pack a GT EVM record
#

import FWCore.ParameterSet.Config as cms

# process
process = cms.Process("TestGtEvmPacker")

# number of events to be processed and source file
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/g/ghete/scratch0/CmsswTestFiles/testGt_EvmPacker_source.root')
)

# load and configure modules

# L1 EventSetup
process.load("L1Trigger.Configuration.L1DummyConfig_cff")

# L1 GT EvmPacker
process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtEvmPack_cfi")
 
# input tag for GT readout record: 
#     gtDigis   = GT emulator (default)
#     l1GtEvmUnpack   = GT EVM unpacker 
#process.l1GtEvmPack.EvmGtInputTag = 'l1GtEvmUnpack'
    

#/ mask for active boards (actually 16 bits)
#      if bit is zero, the corresponding board will not be packed
#      default: no board masked: ActiveBoardsMask = 0xFFFF

# no board masked (default)
#process.l1GtEvmPack.ActiveBoardsMask = 0xFFFF
    
# GTFE only in the record
#process.l1GtEvmPack.ActiveBoardsMask = 0x0000

# GTFE + TCS 
#process.l1GtEvmPack.ActiveBoardsMask = 0x0001

# GTFE + FDL 
#process.l1GtEvmPack.ActiveBoardsMask = 0x0002
     
# GTFE + TCS + FDL
#process.l1GtEvmPack.ActiveBoardsMask = 0x0003

# path to be run
process.p = cms.Path(process.l1GtEvmPack)

# services

# Message Logger
# uncomment / comment messages with DEBUG mode to run in DEBUG mode
process.MessageLogger = cms.Service("MessageLogger",
    testGt_EvmPacker = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),       ## DEBUG mode 

        DEBUG = cms.untracked.PSet( 
            limit = cms.untracked.int32(-1)              ## DEBUG mode, all messages  
        ),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    debugModules = cms.untracked.vstring('l1GtEvmPack'), ## DEBUG mode 
    destinations = cms.untracked.vstring('testGt_EvmPacker')
)

# output 
process.outputL1GtEvmPack = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testGt_EvmPacker_output.root'),
    # keep only packed data in the ROOT file
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_l1GtEvmPack_*_*')
)

process.outpath = cms.EndPath(process.outputL1GtEvmPack)

