#
# customization fragment to run L1 emulator starting from a RAW file
#
# V.M. Ghete 2010-06-09

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
    # print the L1 trigger report
    # comment/un-comment the corresponding flag
    #
    #printL1TriggerReport = False
    printL1TriggerReport = True
    
    if printL1TriggerReport == True :
        from L1Trigger.Configuration.L1Trigger_custom import customiseL1TriggerReport
        process=customiseL1TriggerReport(process)
        
        process.CaloTPG_SimL1Emulator_L1TriggerReport = cms.Sequence(
                                                    process.CaloTPG_SimL1Emulator*process.l1GtTrigReport)
        process.L1simulation_step.replace(
                            process.CaloTPG_SimL1Emulator,process.CaloTPG_SimL1Emulator_L1TriggerReport)

        process.l1GtTrigReport.L1GtRecordInputTag = "simGtDigis"


    #
    return (process)
# foo bar baz
# aArrwETsSDgTB
# ZuppVzdkU0srF
