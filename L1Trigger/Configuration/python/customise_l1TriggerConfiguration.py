#
# special configuration cases for L1 trigger masks, prescale factors and L1 menu 
#
# comment/un-comment the corresponding flag
#
#
# V.M. Ghete 2010-06-10

import FWCore.ParameterSet.Config as cms

def customiseL1TriggerConfiguration(process):
    
    # reset algorithm trigger masks
    
    resetL1GtTriggerMaskAlgoTrig = False
    resetL1GtTriggerMaskAlgoTrig = True
    
    if resetL1GtTriggerMaskAlgoTrig == True :
        from L1Trigger.Configuration.L1Trigger_custom import customiseResetMasksAlgoTriggers
        process=customiseResetMasksAlgoTriggers(process)

    # reset technical trigger masks
    
    resetL1GtTriggerMaskTechTrig = False
    resetL1GtTriggerMaskTechTrig = True
    
    if resetL1GtTriggerMaskTechTrig == True :
        from L1Trigger.Configuration.L1Trigger_custom import customiseResetMasksTechTriggers
        process=customiseResetMasksTechTriggers(process)
        
        
    # reset algorithm trigger veto masks
    
    resetL1GtTriggerMaskVetoAlgoTrig = False
    resetL1GtTriggerMaskVetoAlgoTrig = True
    
    if resetL1GtTriggerMaskVetoAlgoTrig == True :
        from L1Trigger.Configuration.L1Trigger_custom import customiseResetVetoMasksAlgoTriggers
        process=customiseResetVetoMasksAlgoTriggers(process)

    # reset technical trigger veto masks
    
    resetL1GtTriggerMaskVetoTechTrig = False
    resetL1GtTriggerMaskVetoTechTrig = True
    
    if resetL1GtTriggerMaskVetoTechTrig == True :
        from L1Trigger.Configuration.L1Trigger_custom import customiseResetVetoMasksTechTriggers
        process=customiseResetVetoMasksTechTriggers(process)

        

    
    # unprescale algorithm triggers (all prescale factors set to 1)
    
    # temporary solution
    unprescaleL1GtAlgoTriggers = False
    unprescaleL1GtAlgoTriggers = True
    
    if unprescaleL1GtAlgoTriggers == True :
        from L1Trigger.Configuration.L1Trigger_custom import customiseUnprescaleAlgoTriggers
        process=customiseUnprescaleAlgoTriggers(process)

    # unprescale technical triggers (all prescale factors set to 1)
    
    # temporary solution
    unprescaleL1GtTechTriggers = False
    unprescaleL1GtTechTriggers = True
    
    if unprescaleL1GtTechTriggers == True :
        from L1Trigger.Configuration.L1Trigger_custom import customiseUnprescaleTechTriggers
        process=customiseUnprescaleTechTriggers(process)



    # overwrite the L1 trigger menu
    
    overwriteL1Menu = False
    #overwriteL1Menu = True
    
    if overwriteL1Menu == True :
        from L1Trigger.Configuration.L1Trigger_custom import customiseL1Menu
        process=customiseL1Menu(process)
        
        
 
    #
    return (process)
