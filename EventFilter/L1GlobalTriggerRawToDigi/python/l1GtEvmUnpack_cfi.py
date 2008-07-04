# unpack L1 Global Trigger EVM record 

import FWCore.ParameterSet.Config as cms

# unpack L1 Global Trigger EVM record 
l1GtEvmUnpack = cms.EDProducer("L1GlobalTriggerEvmRawToDigi",
                               
    # input tag for GT EVM record: 
    #     source = hardware record, 
    #     l1GtEvmPack = GT EVM packer (DigiToRaw) 
    EvmGtInputTag = cms.InputTag("l1GtEvmPack"),
    
    # FED Id for GT EVM record 
    # default value defined in DataFormats/FEDRawData/src/FEDNumbering.cc
    EvmGtFedId = cms.untracked.int32(812),
    
    # mask for active boards (actually 16 bits)
    #      if bit is zero, the corresponding board will not be unpacked
    #      default: no board masked
    ActiveBoardsMask = cms.uint32(0xFFFF),

    # number of "bunch crossing in the event" (bxInEvent) to be unpacked
    # symmetric around L1Accept (bxInEvent = 0):
    #    1 (bxInEvent = 0); 3 (F 0 1) (standard record); 5 (E F 0 1 2) (debug record)
    # even numbers (except 0) "rounded" to the nearest lower odd number
    # negative value: unpack all available bxInEvent   
    # if more bxInEvent than available are required, unpack what exists and write a warning  
    UnpackBxInEvent = cms.int32(-1),
    
    # length of BST message (in bytes)
    # if negative, take it from event setup
    BstLengthBytes = cms.int32(-1)
     
)


