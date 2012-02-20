#!/usr/bin/env python
"""
_AlCaTestEnable_

Scenario supporting proton collisions

"""

from Configuration.DataProcessing.Scenario import AlCa

class AlCaTestEnable(AlCa):
    """
    _AlCaTestEnable_

    Implement configuration building for data processing for proton
    collision data taking

    """

    def promptReco(self, globalTag, writeTiers = ['ALCARECO'], **args):
        """
        _promptReco_

        Proton collision data taking prompt reco

        """
        skims = ['TkAlLAS']
        return self.promptRecoImpl( globalTag, skims, writeTiers, args)
