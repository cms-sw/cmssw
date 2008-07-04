#
# cfg file to pack a GT DAQ record
#

import FWCore.ParameterSet.Config as cms

# process
process = cms.Process("TestGtPacker")

# number of events to be processed and source file
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/g/ghete/scratch0/CmsswTestFiles/testGt_Packer_source.root')
)

# load and configure modules

# L1 EventSetup
process.load("L1Trigger.Configuration.L1DummyConfig_cff")

# L1 GT Packer
process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtPack_cfi")

# input tag for GT readout record: 
#     gtDigis         = GT emulator (default)
#     l1GtUnpack      = GT unpacker 

#process.l1GtPack.DaqGtInputTag = 'l1GtUnpack'
    
# input tag for GMT readout collection: 
#     gtDigis        = GMT emulator (default)
#     l1GtUnpack     = GT unpacker 

#process.l1GtPack.MuGmtInputTag = 'l1GtUnpack'

# mask for active boards (actually 16 bits)
#      if bit is zero, the corresponding board will not be packed
#      default: no board masked: ActiveBoardsMask = 0xFFFF

# no board masked (default)
#process.l1GtPack.ActiveBoardsMask = 0xFFFF
    
# GTFE only in the record
#process.l1GtPack.ActiveBoardsMask = 0x0000

# GTFE + FDL 
#process.l1GtPack.ActiveBoardsMask = 0x0001
     
# GTFE + GMT 
#process.l1GtPack.ActiveBoardsMask = 0x0100

# GTFE + FDL + GMT 
#process.l1GtPack.ActiveBoardsMask = 0x0101

# path to be run
process.p = cms.Path(process.l1GtPack)

# services

# Message Logger
# uncomment / comment messages with DEBUG mode to run in DEBUG mode
process.MessageLogger = cms.Service("MessageLogger",
    testGt_Packer = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),    ## DEBUG mode 

        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)           ## DEBUG mode, all messages  
        ),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    debugModules = cms.untracked.vstring('l1GtPack'), ## DEBUG mode 
    destinations = cms.untracked.vstring('testGt_Packer')
)

# output 
process.outputL1GtPack = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testGt_Packer_output.root'),
    # keep only packed data in the ROOT file
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_l1GtPack_*_*')
)

process.outpath = cms.EndPath(process.outputL1GtPack)

