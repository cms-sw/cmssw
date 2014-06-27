import FWCore.ParameterSet.Config as cms

mp7BufferDumpToRaw = cms.EDProducer(
    "l1t::MP7BufferDumpToRaw",
    rxFile          = cms.untracked.string("rx_summary.txt"),
    txFile          = cms.untracked.string("tx_summary.txt"),
    fedId           = cms.untracked.int32(1),
    nHeaders        = cms.untracked.int32(3),
    nFramesPerEvent = cms.untracked.int32(41),
    nRxLinks        = cms.untracked.int32(72),
    nTxLinks        = cms.untracked.int32(72),
    rxBlockLength   = cms.untracked.vint32(41,41,41,41,41,41,41,41,41,
                                           41,41,41,41,41,41,41,41,41,
                                           41,41,41,41,41,41,41,41,41,
                                           41,41,41,41,41,41,41,41,41,
                                           41,41,41,41,41,41,41,41,41,
                                           41,41,41,41,41,41,41,41,41,
                                           41,41,41,41,41,41,41,41,41,
                                           41,41,41,41,41,41,41,41,41),
    txBlockLength   = cms.untracked.vint32(12,4,12,8,0,0,0,0,0,
                                           0,0,0,0,0,0,0,0,0,
                                           0,0,0,0,0,0,0,0,0,
                                           0,0,0,0,0,0,0,0,0,
                                           0,0,0,0,0,0,0,0,0,
                                           0,0,0,0,0,0,0,0,0,
                                           0,0,0,0,0,0,0,0,0,
                                           0,0,0,0,0,0,0,0,0)

)
