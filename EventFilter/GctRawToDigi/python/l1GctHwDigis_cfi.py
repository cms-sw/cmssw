import FWCore.ParameterSet.Config as cms

l1GctHwDigis = cms.EDProducer("GctRawToDigi",
    inputLabel = cms.InputTag("rawDataCollector"),
    gctFedId = cms.untracked.int32(745),
    hltMode = cms.bool(False),
    numberOfGctSamplesToUnpack = cms.uint32(1), 
    numberOfRctSamplesToUnpack = cms.uint32(1),
    unpackSharedRegions = cms.bool(False),
    unpackerVersion = cms.uint32(0), #  ** SEE BELOW FOR DETAILS OF THIS OPTION **
    verbose = cms.untracked.bool(False)
)

# Details of "unpackerVersion" option:
# 
#   value   |                        Unpacker/RAW Format Version 
#-----------|---------------------------------------------------------------------------------
#     0     |   Auto-detects RAW Format in use - the recommended option.
#     1     |   Force usage of the Monte-Carlo Legacy unpacker (unpacks DigiToRaw events).
#     2     |   Force usage of the RAW Format V35 unpacker.
#     3     |   Force usage of the RAW Format V38 unpacker.
