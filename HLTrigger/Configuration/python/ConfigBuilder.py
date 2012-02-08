# mapping with the available HLT tables supported by cmsDriver.py
__triggerTable = {

    # HLT GRun trigger table running offline
    'GRun': ( 
        'HLTrigger/Configuration/HLT_GRun_cff', 
    ),

    # HLT GRun trigger table running online
    'GRun_data': ( 
        'HLTrigger/Configuration/HLT_GRun_cff', 
    ),

    # HLT HIon trigger table running offline
    'HIon': ( 
        'HLTrigger/Configuration/HLT_HIon_cff', 
    ),

    # HLT HIon trigger table running online
    'HIon_data': ( 
        'HLTrigger/Configuration/HLT_HIon_cff', 
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

