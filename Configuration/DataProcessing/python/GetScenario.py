#!/usr/bin/env python
"""
_GetScenario_

Util to retrieve a Scenario implementation
Searches Impl directory for the named scenario and imports it


"""


def getScenario(scenarioName):
    """
    _getScenario_

    Util to load the scenario implementation.

    Assumes that module exists at:

    Configuration.DataProcessing.Impl.<scenarioName>.py
    
    """
    moduleName = "Configuration.DataProcessing.Impl.%s" % scenarioName
    try:
        module = __import__(moduleName,
                            globals(), locals(), [scenarioName])#, -1)
    except ImportError, ex:
        msg = "Unable to load Scenario Module:\n"
        msg += "%s\n%s\n" % (moduleName, str(ex))
        raise RuntimeError, msg
    instance = getattr(module, scenarioName, None)
    if instance == None:
        msg = "Unable to retrieve instance of Scenario class:"
        msg += "%s\n From Module\n%s" % (scenarioName, moduleName)
    return instance()


