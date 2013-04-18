#
# customization fragment to run L1 Pattern Generator starting from a RAW file
#
#  - run ECAL and HCAL TPG
#  - run the L1 emulator with a different L1 configuration
#  - run the L1 trigger report
#  - run the pattern generator
#  
# V.M. Ghete 2011-02-10

import FWCore.ParameterSet.Config as cms

def customise(process):
    
    #
    # (re-)run the  L1 emulator starting from a RAW file
    #
    from L1Trigger.Configuration.L1Trigger_custom import customiseL1EmulatorFromRaw
    process=customiseL1EmulatorFromRaw(process)
   
    #
    # special configuration cases (change to desired configuration in customize_l1TriggerConfiguration)
    #
    from L1Trigger.Configuration.customise_l1TriggerConfiguration import customiseL1TriggerConfiguration
    process=customiseL1TriggerConfiguration(process)
 
    #
    # customization of output commands
    #
    from L1Trigger.Configuration.L1Trigger_custom import customiseOutputCommands
    process=customiseOutputCommands(process)
 
    #
    # load and configure the pattern test generator
    #
    process.load("L1Trigger.GlobalTriggerAnalyzer.l1GtPatternGenerator_cfi")

    # take GCT and GMT data from emulators, not from unpackers
    gctLabel = 'simGctDigis'
    gmtLabel = 'simGmtDigis'
    process.l1GtPatternGenerator.GctInputTag = gctLabel
    process.l1GtPatternGenerator.GmtInputTag = gmtLabel

    process.l1GtPatternGenerator.CscInputTag = cms.InputTag("simDttfDigis","DT")
    process.l1GtPatternGenerator.DtInputTag = cms.InputTag("simCsctfDigis","CSC")
    process.l1GtPatternGenerator.RpcbInputTag = cms.InputTag("simRpcTriggerDigis","RPCb")
    process.l1GtPatternGenerator.RpcfInputTag = cms.InputTag("simRpcTriggerDigis","RPCf")

    ## enable detailed output (event no/bx per line, item tracing)
    #process.l1GtPatternGenerator.DebugOutput = cms.bool(True)

    #
    # Global Trigger emulator configuration 
    # input data from the same sources as the pattern writer
    #
    process.simGtDigis.ProduceL1GtEvmRecord = False
    process.simGtDigis.ProduceL1GtObjectMapRecord = False
    process.simGtDigis.WritePsbL1GtDaqRecord = False
    process.simGtDigis.EmulateBxInEvent = -1
         
    #     
    # L1 trigger report
    #
    from L1Trigger.Configuration.L1Trigger_custom import customiseL1TriggerReport
    process=customiseL1TriggerReport(process)
    process.l1GtTrigReport.L1GtRecordInputTag = "simGtDigis"

    #     
    # full sequence pattern generator from raw data
    #
        
    process.LGtPatternGeneratorFromRaw= cms.Sequence(
                        process.CaloTPG_SimL1Emulator*process.l1GtTrigReport*process.l1GtPatternGenerator)
    process.L1simulation_step.replace(
                        process.CaloTPG_SimL1Emulator,process.LGtPatternGeneratorFromRaw)



    #
    return (process)
