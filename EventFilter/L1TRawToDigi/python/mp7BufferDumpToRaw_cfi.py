import FWCore.ParameterSet.Config as cms

mp7BufferDumpToRaw = cms.EDProducer(
    "l1t::MP7BufferDumpToRaw",
    rxFile           = cms.untracked.string("rx_summary.txt"),
    txFile           = cms.untracked.string("tx_summary.txt"),

    # buffer dump file input parameters
    nTextHeaderLines = cms.untracked.int32(3),
    nFramesPerEvent  = cms.untracked.int32(41),
    txLatency        = cms.untracked.int32(54),
    nRxLinks         = cms.untracked.int32(72),
    nTxLinks         = cms.untracked.int32(72),
    nRxEventHeaders  = cms.untracked.int32(0),
    nTxEventHeaders  = cms.untracked.int32(0),

    # DAQ parameters
    fedId            = cms.untracked.int32(2),
    eventType        = cms.untracked.int32(238),
    fwVersion        = cms.untracked.int32(255),
    lenSlinkHeader   = cms.untracked.int32(16),  # length in 8 bit words !
    lenSlinkTrailer  = cms.untracked.int32(8),
    lenAMC13Header   = cms.untracked.int32(0),
    lenAMC13Trailer  = cms.untracked.int32(0),
    lenAMCHeader     = cms.untracked.int32(12),
    lenAMCTrailer    = cms.untracked.int32(8),

    # these parameters specify the amount of data from each link to be
    # recorded in the FEDRawData object
    # if insufficient data is read from any channel to produce the
    # record, module will pad with zeros
    rxBlockLength    = cms.untracked.vint32(41,41,41,41,41,41,41,41,41,
                                            41,41,41,41,41,41,41,41,41,
                                            41,41,41,41,41,41,41,41,41,
                                            41,41,41,41,41,41,41,41,41,
                                            41,41,41,41,41,41,41,41,41,
                                            41,41,41,41,41,41,41,41,41,
                                            41,41,41,41,41,41,41,41,41,
                                            41,41,41,41,41,41,41,41,41),
    txBlockLength    = cms.untracked.vint32(39,39,39,39,39,39,0,0,0,
                                            0,0,0,0,0,0,0,0,0,
                                            0,0,0,0,0,0,0,0,0,
                                            0,0,0,0,0,0,0,0,0,
                                            0,0,0,0,0,0,0,0,0,
                                            0,0,0,0,0,0,0,0,0,
                                            0,0,0,0,0,0,0,0,0,
                                            0,0,0,0,0,0,0,0,0)

)
