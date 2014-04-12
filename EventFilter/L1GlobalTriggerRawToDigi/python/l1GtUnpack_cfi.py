import FWCore.ParameterSet.Config as cms

l1GtUnpack = cms.EDProducer("L1GlobalTriggerRawToDigi",
                            
    # input tag for GT readout collection: 
    #     source = hardware record, 
    #     l1GtPack = GT packer (DigiToRaw) 
    DaqGtInputTag = cms.InputTag("l1GtPack"),

    # FED Id for GT DAQ record 
    # default value defined in DataFormats/FEDRawData/src/FEDNumbering.cc
    DaqGtFedId = cms.untracked.int32(813),

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
    UnpackBxInEvent = cms.int32(-1)

)


