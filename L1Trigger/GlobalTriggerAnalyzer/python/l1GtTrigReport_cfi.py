import FWCore.ParameterSet.Config as cms

l1GtTrigReport = cms.EDAnalyzer("L1GtTrigReport",
                                
    # boolean flag to select the input record
    # if true, it will use L1GlobalTriggerRecord 
    UseL1GlobalTriggerRecord = cms.bool(False),

    # input tag for GT record: 
    #   GT emulator:    gtDigis (DAQ record)
    #   GT unpacker:    gtDigis (DAQ record)
    #   GT lite record: l1GtRecord 
    L1GtRecordInputTag = cms.InputTag("gtDigis"),

    # print verbosity
    #   Level 0 Physics Partition:  TriggerKey, AlgorithmKey, Passed, Rejected, Error
    #   Level 1 Physics Partition:  Level 0 + PrescaleFactors + Mask
    #   Level 2 Physics Partition:  Level 0, Efficiency
    #
    #   Level 10 Physics Partition: TriggerKey, AlgorithmKey, PrescaleFactors
    #                               Before masks: Passed, Rejected, 
    #                               Mask
    #                               After masks:  Passed, Rejected,
    #                               Error
    #
    #   Level 100 All Partitions:   TriggerKey, AlgorithmKey, Passed, Rejected, Error
    #   Level 101 All Partitions:   Level 100 + PrescaleFactors + Mask
    #   ... 
    PrintVerbosity = cms.untracked.int32(2),
    
    # print output
    #   0 std::cout
    #   1 LogTrace
    #   2 LogVerbatim
    #   3 LogInfo
    PrintOutput = cms.untracked.int32(3)
    
)


