import FWCore.ParameterSet.Config as cms

source = cms.Source("FedRawDataInputSource",
    getLSFromFilename = cms.untracked.bool(True),
    eventChunkSize = cms.untracked.uint32(128),
    eventChunkBlock = cms.untracked.uint32(128),
    numBuffers = cms.untracked.uint32(1),
    verifyAdler32 = cms.untracked.bool(True)
    )

