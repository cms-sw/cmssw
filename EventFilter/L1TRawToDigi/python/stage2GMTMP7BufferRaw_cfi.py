import FWCore.ParameterSet.Config as cms

stage2GMTRaw = cms.EDProducer(
    "MP7BufferDumpToRaw",
    rxFile           = cms.untracked.string("rx_summary.txt"),
    txFile           = cms.untracked.string("tx_summary.txt"),

    # input file type
    packetisedData   = cms.untracked.bool(False),

    # parameters for non-packetised data
    nFramesPerEvent  = cms.untracked.int32(6),
    nFramesOffset    = cms.untracked.vuint32(5),
    nFramesLatency   = cms.untracked.vuint32(41),

    # DAQ parameters
    boardId          = cms.untracked.vint32(0),
    fedId            = cms.untracked.int32(1402),
    eventType        = cms.untracked.int32(238),
    fwVersion        = cms.untracked.int32(255),
    lenSlinkHeader   = cms.untracked.int32(8),  # length in 8 bit words !
    lenSlinkTrailer  = cms.untracked.int32(8),
    #mux = cms.untracked.bool(True),

    # these parameters specify the amount of data from each link to be
    # recorded in the FEDRawData object
    # if insufficient data is read from any channel to produce the
    # record, module will pad with zeros
    blocks           = cms.untracked.VPSet(
        cms.untracked.PSet(  # region board setup
            rxBlockLength    = cms.untracked.vint32(0,0,0,0, # q0 0-3
                                                    0,0,0,0, # q1 4-7
                                                    0,0,0,0, # q2 8-11
                                                    0,0,0,0, # q3 12-15
                                                    0,0,0,0, # q4 16-19
                                                    0,0,0,0, # q5 20-23
                                                    0,0,0,0, # q6 24-27
                                                    0,0,0,0, # q7 28-31
                                                    0,0,0,0, # q8 32-35
                                                    6,6,6,6, # q9 36-39
                                                    6,6,6,6, # q10 40-43
                                                    6,6,6,6, # q11 44-47
                                                    6,6,6,6, # q12 48-51
                                                    6,6,6,6, # q13 52-55
                                                    6,6,6,6, # q14 56-59
                                                    6,6,6,6, # q15 60-63
                                                    6,6,6,6, # q16 64-67
                                                    6,6,6,6) # q17 68-71
            ,
            txBlockLength    = cms.untracked.vint32(6,6,6,6) # q0 0-3
            ),
        )
)
