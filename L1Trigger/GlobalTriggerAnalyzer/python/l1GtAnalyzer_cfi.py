import FWCore.ParameterSet.Config as cms

from L1Trigger.GlobalTriggerAnalyzer.L1ExtraInputTagSet_cff import *

l1GtAnalyzer = cms.EDAnalyzer("L1GtAnalyzer",
         
    # non-keyed parameter sets first                           
    # input tags for L1Extra collections
    # used by: 
    #    analyzeTrigger
    L1ExtraInputTagSet,
    
    # print output
    #   0 std::cout
    #   1 LogTrace
    #   2 LogVerbatim
    #   3 LogInfo
    PrintOutput = cms.untracked.int32(3),

    # enable/disable various analyses
    #
    analyzeDecisionReadoutRecordEnable = cms.bool(False),
    #
    analyzeL1GtUtilsMenuLiteEnable = cms.bool(False),
    analyzeL1GtUtilsEventSetupEnable = cms.bool(False),
    analyzeL1GtUtilsEnable = cms.bool(False),
    analyzeTriggerEnable = cms.bool(False),
    #
    analyzeObjectMapEnable = cms.bool(False),
    #
    analyzeL1GtTriggerMenuLiteEnable = cms.bool(False),
    #
    analyzeConditionsInRunBlockEnable = cms.bool(False),
    analyzeConditionsInLumiBlockEnable = cms.bool(False),
    analyzeConditionsInEventBlockEnable = cms.bool(False),

    # input tag for L1GlobalTriggerReadoutRecord (L1 GT DAQ readout record) 
    #     GT emulator, GT unpacker:  gtDigis  
    # used by: 
    #    analyzeDecisionReadoutRecord
    #    L1GtUtils methods with input tags explicitly given
    #        analyzeL1GtUtilsEventSetup
    #        analyzeL1GtUtils
    #        analyzeTrigger
    L1GtDaqReadoutRecordInputTag = cms.InputTag("gtDigis"),
    
    # input tag for L1GlobalTriggerRecord 
    #     L1GlobalTriggerRecord record producer:  l1GtRecord  
    # used by: 
    #    analyzeDecisionReadoutRecord
    #    L1GtUtils methods with input tags explicitly given, if L1GtDaqReadoutRecordInputTag product does not exist
    #        analyzeL1GtUtilsEventSetup
    #        analyzeL1GtUtils
    #        analyzeTrigger
    L1GtRecordInputTag = cms.InputTag("l1GtRecord"),
    
    # input tag for GT object map collection L1GlobalTriggerObjectMapRecord
    #     only the L1 GT emulator produces it,
    #     no map collection is produced by hardware
    #    
    #     GT emulator:  gtDigis
    # used by: 
    #    analyzeObjectMap
    #    analyzeTrigger
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),

    # input tag for GT object map collection L1GlobalTriggerObjectMaps
    #     no map collection is produced by hardware
    #    
    #     L1Reco producer:  l1L1GtObjectMap
    # used by: 
    #    analyzeObjectMap
    #    analyzeTrigger
    L1GtObjectMapsInputTag = cms.InputTag("l1L1GtObjectMap"),

    # input tag for GMT readout collection: not used
    #     gmtDigis = GMT emulator
    #     gtDigis  = GT unpacker (common GT/GMT unpacker)
    L1GmtInputTag = cms.InputTag("gmtDigis"),
    
    # input tag for L1GtTriggerMenuLite
    # used by: 
    #    analyzeL1GtTriggerMenuLite (always)
    #    L1GtUtils methods, if L1GtTmLInputTagProv is set to False    
    #        analyzeL1GtUtilsMenuLite
    #        analyzeL1GtUtils
    #        analyzeTrigger
    L1GtTmLInputTag = cms.InputTag("l1GtTriggerMenuLite"),
    
    # input tag for input tag for ConditionInEdm products
    # used by: 
    #    analyzeConditionsInRunBlock
    #    analyzeConditionsInLumiBlock
    #    analyzeConditionsInEventBlock    
    CondInEdmInputTag = cms.InputTag("conditionsInEdm"),

    # an algorithm trigger name or alias, or a technical trigger name
    # used by: 
    #    analyzeL1GtUtilsEventSetup
    #    analyzeL1GtUtilsMenuLite
    #    analyzeL1GtUtils
    #    analyzeTrigger
    #    analyzeObjectMap (relevant for algorithm triggers only)
    AlgorithmName = cms.string('L1_SingleEG20'),
    # 
    #  a condition in the above algorithm trigger
    # used by: 
    #    analyzeObjectMap (relevant for algorithm triggers only)
    ConditionName = cms.string('SingleIsoEG_0x14'),
    
    # a bit number for an algorithm trigger or a technical trigger
    # special use - do never use a bit number in an analysis
    # used by: 
    #    analyzeL1GtTriggerMenuLite
    BitNumber = cms.uint32(0),
    
    # select for L1GtUtils methods the L1 configuration to use: 0, 100000, 200000
    # used by: 
    #    analyzeL1GtUtilsEventSetup
    #    analyzeL1GtUtilsMenuLite
    #    analyzeL1GtUtils
    #    analyzeTrigger
    L1GtUtilsConfiguration = cms.uint32(0),
    
    # if true, use methods in L1GtUtils with the input tag for L1GtTriggerMenuLite
    # from provenance
    # used by: 
    #    analyzeL1GtUtilsEventSetup
    #    analyzeL1GtUtilsMenuLite
    #    analyzeL1GtUtils
    #    analyzeTrigger
    L1GtTmLInputTagProv = cms.bool(True),
 
    # if true, use methods in L1GtUtils with the given input tags 
    # for L1GlobalTriggerReadoutRecord and / or L1GlobalTriggerRecord from provenance
    # used by: 
    #    analyzeL1GtUtilsEventSetup
    #    analyzeL1GtUtilsMenuLite
    #    analyzeL1GtUtils
    #    analyzeTrigger
    L1GtRecordsInputTagProv = cms.bool(True),

    # if true, configure (partially) L1GtUtils in beginRun using getL1GtRunCache
    # used by: 
    #    analyzeL1GtUtilsEventSetup
    #    analyzeL1GtUtilsMenuLite
    #    analyzeL1GtUtils
    #    analyzeTrigger
    L1GtUtilsConfigureBeginRun = cms.bool(True)
   
)


