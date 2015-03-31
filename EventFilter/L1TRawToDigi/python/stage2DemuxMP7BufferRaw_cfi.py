
import FWCore.ParameterSet.Config as cms

stage2DemuxRaw = cms.EDProducer(
    "MP7BufferDumpToRaw",
    rxFile           = cms.untracked.string("demux_rx_summary.txt"),
    txFile           = cms.untracked.string("demux_tx_summary.txt"),

    # input file type
    packetisedData   = cms.untracked.bool(False),

    # parameters for non-packetised data
    nFramesPerEvent  = cms.untracked.int32(6),
    nFramesOffset    = cms.untracked.vuint32(0),
    nFramesLatency   = cms.untracked.vuint32(0),

    # DAQ parameters
    fedId            = cms.untracked.int32(1366),
    eventType        = cms.untracked.int32(238),
    fwVersion        = cms.untracked.int32(255),
    lenSlinkHeader   = cms.untracked.int32(8),  # length in 8 bit words !
    lenSlinkTrailer  = cms.untracked.int32(8),

    # readout parameters
    boardId          = cms.untracked.vint32( 0 ),
    mux              = cms.untracked.bool(False),
    muxOffset        = cms.untracked.int32(0),

    # these parameters specify the amount of data from each link to be
    # recorded in the FEDRawData object
    # if insufficient data is read from any channel to produce the
    # record, module will pad with zeros
    blocks           = cms.untracked.VPSet(
        cms.untracked.PSet(
            rxBlockLength    = cms.untracked.vint32(0,0,0,0, # q0 0-3
                                                    0,0,0,0, # q1 4-7
                                                    0,0,0,0, # q2 8-11
                                                    0,0,0,0, # q3 12-15
                                                    0,0,0,0, # q4 16-19
                                                    0,0,0,0, # q5 20-23
                                                    0,0,0,0, # q6 24-27
                                                    0,0,0,0, # q7 28-31
                                                    0,0,0,0, # q8 32-35
                                                    0,0,0,0, # q9 36-39
                                                    0,0,0,0, # q10 40-43
                                                    0,0,0,0, # q11 44-47
                                                    0,0,0,0, # q12 48-51
                                                    0,0,0,0, # q13 52-55
                                                    0,0,0,0, # q14 56-59
                                                    0,0,0,0, # q15 60-63
                                                    0,0,0,0, # q16 64-67
                                                    0,0,0,0), # q17 68-71
            
            txBlockLength    = cms.untracked.vint32(0,0,0,0, # q0 0-3
                                                    0,0,6,6, # q1 4-7
                                                    0,0,6,0, # q2 8-11
                                                    0,0,0,0, # q3 12-15
                                                    0,0,0,0, # q4 16-19
                                                    0,0,0,0, # q5 20-23
                                                    0,0,0,0, # q6 24-27
                                                    0,0,0,0, # q7 28-31
                                                    0,0,0,0, # q8 32-35
                                                    0,0,0,0, # q9 36-39
                                                    0,0,0,0, # q10 40-43
                                                    0,0,0,0, # q11 44-47
                                                    0,0,0,0, # q12 48-51
                                                    0,0,0,0, # q13 52-55
                                                    0,0,0,0, # q14 56-59
                                                    0,0,0,0, # q15 60-63
                                                    0,0,0,0, # q16 64-67
                                                    0,0,0,0) # q17 68-71
            )
        )

)
