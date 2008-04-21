import FWCore.ParameterSet.Config as cms

l1GtAnalyz = cms.EDAnalyzer("L1GtAnalyzer",
    # input tag for GT readout collection: 
    #     GT emulator:  gtDigis  
    #     GT unpacker:  l1GtUnpack  
    DaqGtInputTag = cms.InputTag("gtDigis"),
    #
    # input tags for GT lite record
    #     L1 GT lite record producer:  l1GtRecord  
    L1GtRecordInputTag = cms.InputTag("l1GtRecord"),
    #
    # input tag for GMT readout collection: 
    #     gmtDigis = GMT emulator
    #     l1GtUnpack     = GT unpacker (common GT/GMT unpacker)
    GmtInputTag = cms.InputTag("gmtDigis"),
    #
    # input tag for GT object map collection
    #     only the L1 GT emulator produces it,
    #     no map collection is produced by hardware
    #
    #     GT emulator:  gtDigis  
    GtObjectMapTag = cms.InputTag("gtDigis"),
    #
    # an algorithm and a condition in that algorithm to test the object maps
    ConditionName = cms.string('DoubleTauJet50_2'),
    AlgorithmName = cms.string('L1_DoubleJet50_ETM20')
)


