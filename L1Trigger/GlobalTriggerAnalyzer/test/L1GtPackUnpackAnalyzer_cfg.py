#
# cfg file to pack (DigiToRaw) a GT DAQ record, unpack (RawToDigi) it back
# and compare the two set of digis
#

import FWCore.ParameterSet.Config as cms

# process
process = cms.Process("TestGtPackUnpackAnalyzer")

# number of events and source
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/g/ghete/scratch0/CmsswTestFiles/testGt_PackUnpackAnalyzer_source.root')
)

# configuration

# L1 EventSetup
process.load("L1Trigger.Configuration.L1DummyConfig_cff")

#
# pack.........
#
process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtPack_cfi")

# input tag for GT readout record: 
process.l1GtPack.DaqGtInputTag = 'gtDigis'
    
# input tag for GMT readout collection: 
process.l1GtPack.MuGmtInputTag = 'hltGtDigis'

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

#
# unpack.......
#

import EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi
process.gtPackedUnpack = EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi.l1GtUnpack.clone()

# input tag for GT and GMT readout collections in the packed data: 
process.gtPackedUnpack.DaqGtInputTag = 'l1GtPack'

# Active Boards Mask
# no board masked (default)
#process.gtPackedUnpack.ActiveBoardsMask = 0xFFFF

# GTFE only in the record
#process.gtPackedUnpack.ActiveBoardsMask = 0x0000

# GTFE + FDL 
#process.gtPackedUnpack.ActiveBoardsMask = 0x0001

# GTFE + GMT 
#process.gtPackedUnpack.ActiveBoardsMask = 0x0100

# GTFE + FDL + GMT 
#process.gtPackedUnpack.ActiveBoardsMask = 0x0101

# BxInEvent to be unpacked

# all available BxInEvent (default)
#process.gtPackedUnpack.UnpackBxInEvent = -1 

# BxInEvent = 0 (L1A)
#process.gtPackedUnpack.UnpackBxInEvent = 1 

# 3 BxInEvent (F, 0, 1)  
#process.gtPackedUnpack.UnpackBxInEvent = 3 

#
# compare the initial and final digis .......
#
process.load("L1Trigger.GlobalTriggerAnalyzer.l1GtPackUnpackAnalyzer_cfi")

# input tag for the initial GT DAQ record: must match the pack label
#process.l1GtPackUnpackAnalyzer.InitialDaqGtInputTag = 'gtDigis'

# input tag for the initial GMT readout collection: must match the pack label 
process.l1GtPackUnpackAnalyzer.InitialMuGmtInputTag = 'hltGtDigis'

# input tag for the final GT DAQ and GMT records:  must match the unpack label 
#     GT unpacker:  gtPackedUnpack (cloned unpacker from L1GtPackUnpackAnalyzer.cfg)
#process.l1GtPackUnpackAnalyzer.FinalGtGmtInputTag = 'gtPackedUnpack'

# path to be run
process.p = cms.Path(process.l1GtPack*process.gtPackedUnpack*process.l1GtPackUnpackAnalyzer)

# services

process.MessageLogger = cms.Service("MessageLogger",
    testGt_PackUnpackAnalyzer = cms.untracked.PSet(
        DEBUG = cms.untracked.PSet( 

            #limit = cms.untracked.int32(-1)          ## DEBUG mode, all messages  
            limit = cms.untracked.int32(10)         ## DEBUG mode, max 10 messages 
        ),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    destinations = cms.untracked.vstring('testGt_PackUnpackAnalyzer'),
    debugModules = cms.untracked.vstring( 'l1GtPack', 'l1GtUnpack', 'l1GtPackUnpackAnalyzer') ## DEBUG mode 
)

# output 
process.outputL1GtPackUnpack = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testGt_PackUnpackAnalyzer_output.root'),
    # keep only emulated data, packed data, unpacked data in the ROOT file
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_gtDigis_*_*', 
        'keep *_gmtDigis_*_*', 
        'keep *_l1GtPack_*_*', 
        'keep *_l1GtPackedUnpack_*_*')
)

process.outpath = cms.EndPath(process.outputL1GtPackUnpack)

