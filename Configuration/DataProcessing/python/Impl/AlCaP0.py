#!/usr/bin/env python
"""
_AlCaP0_

Scenario supporting proton collisions

"""

from Configuration.DataProcessing.AlCa import AlCa

class AlCaP0(AlCa):
    """
    _AlCaP0_

    Implement configuration building for data processing for proton
    collision data taking

    """


    def promptReco(self, globalTag, writeTiers = ['ALCARECO'], **args):
        """
        _promptReco_

        Proton collision data taking prompt reco

        """
        skims = ['EcalCalPi0Calib',
                 'EcalCalEtaCalib']
        return self.promptRecoImpl(globalTag, skims, writeTiers, args)

