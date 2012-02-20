#!/usr/bin/env python
"""
_AlCaPhiSymEcal_

Scenario supporting proton collisions

"""

from Configuration.DataProcessing.AlCa import AlCa

class AlCaPhiSymEcal(AlCa):
    """
    _AlCaPhiSymEcal_

    Implement configuration building for data processing for proton
    collision data taking

    """


    def promptReco(self, globalTag, writeTiers = ['ALCARECO'], **args):
        """
        _promptReco_

        Proton collision data taking prompt reco

        """
        skims = ['EcalCalPhiSym']
        return self.promptRecoImpl(globalTag, skims, writeTiers, args)


