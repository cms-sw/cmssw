#!/usr/bin/env python3
"""
_cosmicsHybrid_

Scenario supporting cosmics data taking in hybrid mode
"""

from Configuration.DataProcessing.Impl.cosmics import cosmics

class cosmicsHybrid(cosmics):
    def __init__(self):
        cosmics.__init__(self)
        self.customs = [ "RecoLocalTracker/SiStripZeroSuppression/customiseHybrid.runOnHybridZS" ]
    """
    _cosmicsHybrid_

    Implement configuration building for data processing for cosmic
    data taking with the strip tracker in hybrid ZS mode

    """

    def promptReco(self, globalTag, **args):
        if not "customs" in args:
            args["customs"] = list(self.customs)
        else:
            args["customs"] += self.customs

        return cosmics.promptReco(self, globalTag, **args)

    def expressProcessing(self, globalTag, **args):
        if not "customs" in args:
            args["customs"] = list(self.customs)
        else:
            args["customs"] += self.customs

        return cosmics.expressProcessing(self, globalTag, **args)

    def visualizationProcessing(self, globalTag, **args):
        if not "customs" in args:
            args["customs"] = list(self.customs)
        else:
            args["customs"] += self.customs

        return cosmics.visualizationProcessing(self, globalTag, **args)
