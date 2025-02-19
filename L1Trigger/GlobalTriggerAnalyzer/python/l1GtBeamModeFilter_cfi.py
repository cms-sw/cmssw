import FWCore.ParameterSet.Config as cms

l1GtBeamModeFilter = cms.EDFilter("L1GtBeamModeFilter",

    # input tag for input tag for ConditionInEdm products
    CondInEdmInputTag = cms.InputTag("conditionsInEdm"),

    # input tag for the L1 GT EVM product 
    L1GtEvmReadoutRecordTag = cms.InputTag("gtEvmDigis"),
    #
    # vector of allowed beam modes 
    # default value: 11 (STABLE)
    AllowedBeamMode = cms.vuint32(11),
    
    # return the inverted result, to be used instead of NOT
    # normal result: true if filter true
    #                false if filter false or error (no product found)
    # inverted result: true if filter false
    #                  false if filter true or error (no product found)
    InvertResult = cms.bool( False )
)
