import FWCore.ParameterSet.Config as cms

gtDigis = cms.EDProducer("L1GlobalTrigger",
    
    # input tag for GMT readout collection: 
    #     gmtDigis = GMT emulator
    #     l1GtUnpack     = GT unpacker (common GT/GMT unpacker)
    GmtInputTag = cms.InputTag("gmtDigis"),
    
    # input tag for GCT readout collections: 
    #     gctDigis = GCT emulator 
    #     l1GctUnpack    = GCT unpacker 
    GctInputTag = cms.InputTag("gctDigis"),
    
    # input tag for CASTOR record 
    #     castorL1Digis =  CASTOR
    CastorInputTag = cms.InputTag("castorL1Digis"),
    
    # technical triggers: a vector of input tags, one tag per each technical 
    # trigger producer 
    # 
    # by default: empty vector
    
    # Example:
    # TechnicalTriggersInputTags = cms.VInputTag(cms.InputTag('aTechTrigDigis'), 
    #                                            cms.InputTag('anotherTechTriggerDigis')),
    TechnicalTriggersInputTags = cms.VInputTag(),
    
    # logical flag to produce the L1 GT DAQ readout record
    #     if true, produce the record
    ProduceL1GtDaqRecord = cms.bool(True),
    
    # logical flag to produce the L1 GT EVM readout record
    #     if true, produce the record
    ProduceL1GtEvmRecord = cms.bool(True),
    
    # logical flag to produce the L1 GT object map record
    #     if true, produce the record
    ProduceL1GtObjectMapRecord = cms.bool(True),
    
    # logical flag to write the PSB content in the  L1 GT DAQ record
    #     if true, write the PSB content in the record
    WritePsbL1GtDaqRecord = cms.bool(True),
    
    # logical flag to read the technical trigger records
    #     if true, it will read via getMany the available records
    ReadTechnicalTriggerRecords = cms.bool(True),
    
    # number of "bunch crossing in the event" (BxInEvent) to be emulated
    # symmetric around L1Accept (BxInEvent = 0):
    #    1 (BxInEvent = 0); 3 (F 0 1) (standard record); 5 (E F 0 1 2) (debug record)
    # even numbers (except 0) "rounded" to the nearest lower odd number
    # negative value: emulate TotalBxInEvent as given in EventSetup  
    EmulateBxInEvent = cms.int32(3),   
    
    # number of BXs in the event corresponding to alternative 0 and 1 in altNrBxBoard()
    # EmulateBxInEvent >= max(RecordLength[0], RecordLength[1])
    # negative values: take the numbers from event setup, from L1GtParameters - NOT AVAILABLE YET
    RecordLength = cms.vint32(3, 0),

    # alternative for number of BX per active board in GT DAQ record: 0 or 1
    # the position is identical with the active board bit
    AlternativeNrBxBoardDaq = cms.uint32(0x0000),  

    # alternative for number of BX per active board in GT EVM record: 0 or 1
    # the position is identical with the active board bit
    AlternativeNrBxBoardEvm = cms.uint32(0x0000),  
    
    # length of BST record (in bytes) from parameter set
    # negative value: take the value from EventSetup      
    BstLengthBytes = cms.int32(-1)
    
)


