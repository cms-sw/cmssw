# mapping with the available HLT tables supported by cmsDriver.py
__triggerTable = {

    # HLT trigger table for the 2009 STARTUP 8E29 menu
    '8E29': ( 
        'FastSimulation/Configuration/HLT_8E29_cff', 
    ),

    # HLT trigger table for the 2009 Global Run menu
    'GRun': (
        'FastSimulation/Configuration/HLT_GRun_cff',
    ),

    # HLT trigger table for the 2009 STARTUP 1E31 menu
    '1E31': ( 
        'FastSimulation/Configuration/HLT_1E31_cff', 
    )
}

# default trigger, used if none is specified
__default = 'GRun'


def getDefaultTrigger():
    """
    Retrieves the default trigger name
    """
    return __default
    

def getHelp():
    """
    Retrieves the list of available triggers
    """
    return __triggerTable.iterkeys()
  

def getConfigsForScenario(sequence = None):
    """
    Retrieves the list of files needed to run a given trigger menu.
    If no trigger or an invalid trigger is given, use the default one. 
    """

    if not sequence:
        # no trigger was specified, use the default one
        trigger = __default
    else:
        # check if the specified trigger is valid
        trigger = sequence
        if trigger not in __triggerTable:
            print 'An unsupported trigger has been requested: %s' % sequence
            print 'The default one will be used instead: %s' % __default
            print 'The supported triggers are:'
            for key in __triggerTable.iterkeys():
                print '\t%s' % key
            print
            trigger = __default

    return __triggerTable[trigger]

