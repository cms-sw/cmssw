import FWCore.ParameterSet.Config as cms

stage1Raw = cms.EDProducer(
    "MP7BufferDumpToRaw",
    rxFile           = cms.untracked.string("rx_summary.txt"),
    txFile           = cms.untracked.string("tx_summary.txt"),

    # input file type
    packetisedData   = cms.untracked.bool(False),

    # parameters for non-packetised data
    nFramesPerEvent  = cms.untracked.int32(6),
    nFramesOffset    = cms.untracked.vint32(0,9),
    nFramesLatency   = cms.untracked.vint32(0,0),

    # DAQ parameters
    boardId          = cms.untracked.vint32( 0x100D, 0x100E ),  # jet ID, EG ID
    fedId            = cms.untracked.int32(1352),
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
            rxBlockLength    = cms.untracked.vint32(6,6,6,6, # quad 0
                                                    6,6,6,6, # quad 1
                                                    6,6,6,6, # quad 2
                                                    6,6,6,6, # quad 3
                                                    6,6,6,6, # quad 4
                                                    6,6,6,6, # quad 5
                                                    6,6,6,6, # quad 6
                                                    6,6,6,6, # quad 7
                                                    6,6,6,6, # quad 8
                                                    0,0,0,0, # quad 9
                                                    0,0,0,0, # quad 10
                                                    0,0,0,0, # quad 11
                                                    0,0,0,0, # quad 12
                                                    0,0,0,0) # quad 13
            ,
            txBlockLength    = cms.untracked.vint32(0,0,0,0, # quad 0
                                                    0,0,0,0, # quad 1
                                                    0,0,0,0, # quad 2
                                                    0,0,0,0, # quad 3
                                                    0,0,0,0, # quad 4
                                                    0,0,0,0, # quad 5
                                                    0,0,0,0, # quad 6
                                                    0,0,0,0, # quad 7
                                                    0,0,0,0, # quad 8
                                                    0,0,2,2, # quad 9
                                                    2,2,2,2, # quad 10
                                                    2,2,2,2, # quad 11
                                                    2,2,2,2, # quad 12
                                                    2,2,0,0) # quad 13
            ),
        cms.untracked.PSet(  # EG board setup
            rxBlockLength    = cms.untracked.vint32(6,6,6,6, # quad 0
                                                    6,6,6,6, # quad 1
                                                    6,6,6,6, # quad 2
                                                    6,6,6,6, # quad 3
                                                    6,6,6,6, # quad 4
                                                    6,6,6,6, # quad 5
                                                    6,6,6,6, # quad 6
                                                    6,6,6,6, # quad 7
                                                    6,6,6,6, # quad 8
                                                    0,0,0,0, # quad 9
                                                    0,0,0,0, # quad 10
                                                    0,0,0,0, # quad 11
                                                    0,0,0,0, # quad 12
                                                    0,0,0,0) # quad 13
            ,
            txBlockLength    = cms.untracked.vint32(0,0,0,0, # quad 0
                                                    0,0,0,0, # quad 1
                                                    0,0,0,0, # quad 2
                                                    0,0,0,0, # quad 3
                                                    0,0,0,0, # quad 4
                                                    0,0,0,0, # quad 5
                                                    0,0,0,0, # quad 6
                                                    0,0,0,0, # quad 7
                                                    0,0,0,0, # quad 8
                                                    0,0,0,0, # quad 9
                                                    0,0,0,0, # quad 10
                                                    0,0,0,0, # quad 11
                                                    0,0,0,0, # quad 12
                                                    0,0,0,0) # quad 13
            )
        )
)
