import FWCore.ParameterSet.Config as cms
import EventFilter.GctRawToDigi.gctRawToDigi_cfi

l1GctHwDigis = EventFilter.GctRawToDigi.gctRawToDigi_cfi.gctRawToDigi.clone()
l1GctHwDigis.inputLabel = cms.InputTag("rawDataCollector")
l1GctHwDigis.gctFedId = cms.untracked.int32(745)
l1GctHwDigis.hltMode = cms.bool(False)
l1GctHwDigis.numberOfGctSamplesToUnpack = cms.uint32(1) 
l1GctHwDigis.numberOfRctSamplesToUnpack = cms.uint32(1)
l1GctHwDigis.unpackSharedRegions = cms.bool(False)
l1GctHwDigis.unpackerVersion = cms.uint32(0)  #  ** SEE BELOW FOR DETAILS OF THIS OPTION **
l1GctHwDigis.verbose = cms.untracked.bool(False)

# Details of "unpackerVersion" option:
# 
#   value   |                        Unpacker/RAW Format Version 
#-----------|---------------------------------------------------------------------------------
#     0     |   Auto-detects RAW Format in use - the recommended option.
#     1     |   Force usage of the Monte-Carlo Legacy unpacker (unpacks DigiToRaw events).
#     2     |   Force usage of the RAW Format V35 unpacker.
#     3     |   Force usage of the RAW Format V38 unpacker.
