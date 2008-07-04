import FWCore.ParameterSet.Config as cms

l1GtEvmPack = cms.EDProducer("L1GTEvmDigiToRaw",
    
    # FED Id for GT EVM record 
    # default value defined in DataFormats/FEDRawData/src/FEDNumbering.cc
    EvmGtFedId = cms.untracked.int32(812),
    
    # input tag for GT readout record: 
    #     gtDigis         = GT emulator, 
    #     l1GtEvmUnpack   = GT EVM unpacker 
    EvmGtInputTag = cms.InputTag("gtDigis"),
    
    # mask for active boards (actually 16 bits)
    #      if bit is zero, the corresponding board will not be packed
    #      default: no board masked
    ActiveBoardsMask = cms.uint32(0xFFFF)
)


