import FWCore.ParameterSet.Config as cms

#
l1GtParameters = cms.ESProducer("L1GtParametersTrivialProducer",
    
    # number of bunch crossing in the GT readout record: 3 bx (standard), 5 bx (debug)
    TotalBxInEvent = cms.int32(3),

    # list of active boards for L1 GT DAQ record (actually 16 bits)
    # default: all active 0xFFFF
    DaqActiveBoards = cms.uint32(0xFFFF),
    
    # list of active boards for L1 GT EVM record (actually 16 bits)
    # default: all active 0xFFFF
    EvmActiveBoards = cms.uint32(0xFFFF),
    
    # number of Bx per board in the DAQ record
    DaqNrBxBoard = cms.vint32(3,
                              3, 3, 3, 3, 3, 3, 3,
                              3),

    # number of Bx per board in the EVM record
    EvmNrBxBoard = cms.vint32(1, 3),
    
    # length of BST record (in bytes) for L1 GT EVM record
    BstLengthBytes = cms.uint32(30)
    
    
)


