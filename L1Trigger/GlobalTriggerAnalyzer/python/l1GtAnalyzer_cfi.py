import FWCore.ParameterSet.Config as cms

l1GtAnalyzer = cms.EDAnalyzer("L1GtAnalyzer",

    # input tag for GT readout collection: 
    #     GT emulator, GT unpacker:  gtDigis  
    L1GtDaqReadoutRecordInputTag = cms.InputTag("gtDigis"),
    
    # input tags for GT lite record
    #     L1 GT lite record producer:  l1GtRecord  
    L1GtRecordInputTag = cms.InputTag("l1GtRecord"),
    
    # input tag for GT object map collection
    #     only the L1 GT emulator produces it,
    #     no map collection is produced by hardware
    
    #     GT emulator:  gtDigis  
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),

    # input tag for GMT readout collection: 
    #     gmtDigis = GMT emulator
    #     gtDigis  = GT unpacker (common GT/GMT unpacker)
    L1GmtInputTag = cms.InputTag("gmtDigis"),
    
    # input tag for L1GtTriggerMenuLite
    L1GtTmLInputTag = cms.InputTag("l1GtTriggerMenuLite"),
    
    # input tag for input tag for ConditionInEdm products
    CondInEdmInputTag = cms.InputTag("conditionsInEdm"),

    # an algorithm and a condition in that algorithm to test the object maps, a bit number
    AlgorithmName = cms.string('L1_DoubleJet50_ETM20'),
    ConditionName = cms.string('DoubleTauJet50_2'),
    BitNumber = cms.uint32(0),
    
    # select the L1 configuration use: 0, 100000, 200000
    L1GtUtilsConfiguration = cms.uint32(0),
    
    # if true, use methods in L1GtUtils with the input tag for L1GtTriggerMenuLite
    # from provenance
    L1GtTmLInputTagProv = cms.bool(True)
    
)


