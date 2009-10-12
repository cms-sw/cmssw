# mapping with the available L1 tables supported by cmsDriver.py
__triggerTable = {
    # L1 conditions and trigger table for the 2009 startup/8E29 menu v0
    'L1Menu_Commissioning2009_v0:Unprescaled': (
        'L1Trigger/Configuration/SimL1Emulator_cff',
        'L1TriggerConfig/L1GtConfigProducers/Luminosity/startup/L1Menu_Commissioning2009_v0_L1T_Scales_20080926_startup_Imp0_Unprescaled_cff'
    ),

    # L1 conditions and trigger table for the 2009 startup/8E29 menu v1
    'L1Menu_Commissioning2009_v1:Unprescaled': (
        'L1Trigger/Configuration/SimL1Emulator_cff',
        'L1TriggerConfig/L1GtConfigProducers/Luminosity/startup/L1Menu_Commissioning2009_v1_L1T_Scales_20080926_startup_Imp0_Unprescaled_cff'
    ),

    # L1 conditions and trigger table for the 2009 startup/8E29 menu v2
    'L1Menu_Commissioning2009_v2:Unprescaled': (
        'L1Trigger/Configuration/SimL1Emulator_cff',
        'L1TriggerConfig/L1GtConfigProducers/Luminosity/startup/L1Menu_Commissioning2009_v2_L1T_Scales_20080926_startup_Imp0_Unprescaled_cff'
    ),

    # L1 conditions and trigger table for the 2009 startup/8E29 menu v3
    'L1Menu_Commissioning2009_v3:Unprescaled': (
        'L1Trigger/Configuration/SimL1Emulator_cff',
        'L1TriggerConfig/L1GtConfigProducers/Luminosity/startup/L1Menu_Commissioning2009_v3_L1T_Scales_20080926_startup_Imp0_Unprescaled_cff'
    ),

    # L1 conditions and trigger table for the 2009 startup/8E29 menu v5
    'L1Menu_Commissioning2009_v5:Unprescaled': (
        'L1Trigger/Configuration/SimL1Emulator_cff',
        'L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_Commissioning2009_v5_L1T_Scales_20080926_startup_Imp0_Unprescaled_cff'
    ),

    # L1 conditions and trigger table for the 2009 1E31 menu v0
    'L1Menu_MC2009_v0:Unprescaled': (
        'L1Trigger/Configuration/SimL1Emulator_cff',
        'L1TriggerConfig/L1GtConfigProducers/Luminosity/lumi1031/L1Menu_MC2009_v0_L1T_Scales_20080922_Imp0_Unprescaled_cff'
    ),

    # L1 conditions and trigger table for the 2009 1E31 menu v1
    'L1Menu_MC2009_v1:Unprescaled': (
        'L1Trigger/Configuration/SimL1Emulator_cff',
        'L1TriggerConfig/L1GtConfigProducers/Luminosity/lumi1031/L1Menu_MC2009_v1_L1T_Scales_20080922_Imp0_Unprescaled_cff'
    ),

    # L1 conditions and trigger table for the 2009 1E31 menu v2 L1T_Scales_20090519
    'L1Menu_MC2009_v2:Unprescaled': (
        'L1Trigger/Configuration/SimL1Emulator_cff',
        'L1TriggerConfig/L1GtConfigProducers/Luminosity/lumi1031/L1Menu_MC2009_v2_L1T_Scales_20090519_Imp0_Unprescaled_cff'
    ),

    # L1 conditions and trigger table for the 2009 1E31 menu v4 L1T_Scales_20090624
    'L1Menu_MC2009_v4:Unprescaled': (
        'L1Trigger/Configuration/SimL1Emulator_cff',
        'L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1031.L1Menu_MC2009_v4_L1T_Scales_20090624_Imp0_Unprescaled_cff'
    ),

    # L1 conditions and trigger table for the 2008 1E30 menu
    'L1Menu_2008MC_2E30:Unprescaled': (
        'L1Trigger/Configuration/SimL1Emulator_cff',
        'L1TriggerConfig/L1GtConfigProducers/Luminosity/lumi1030/L1Menu_2008MC_2E30_Unprescaled_cff'
    ),
 
    # L1 conditions and trigger table for the CRAFT 2008 menu
    'L1Menu_startup2_v4:Unprescaled': (
        'L1Trigger/Configuration/SimL1Emulator_cff',
        'L1TriggerConfig/L1GtConfigProducers/Luminosity/startup/L1Menu_startup2_v4_L1T_Scales_20080926_startup_Imp0_Unprescaled_cff'
    ),
    
    'GlobalTag': (
        'L1Trigger/Configuration/SimL1Emulator_cff',
    )

}

import re

def getConfigsForScenario(sequence = None):
    """
    Retrieves the list of files needed to run a given trigger menu.
    If no trigger or an invalid trigger is given, use the default one. 
    """
    # default trigger, used if none is 
    default = 'GlobalTag'

    if not sequence:
        # no trigger was specified, use the default one
        trigger = default
    else:
        # check if a prescale was specified, and default to 'Unprescaled' if none was given
        args = re.split(r'[,:]', sequence)
        sequence = ':'.join(args)
        if len(args) == 1:
            args.append('Unprescaled')
        trigger = ':'.join(args)

        # check if the specified trigger (augmented with the prescale) is valid
        if trigger not in __triggerTable:
            print 'An unsupported trigger has been requested: %s' % sequence
            print 'The default one will be used instead: %s' % default
            print 'The supported triggers are:'
            for key in __triggerTable.iterkeys():
                print '\t%s' % key
            print
            trigger = default

    return __triggerTable[trigger]

