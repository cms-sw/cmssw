#!/usr/bin/env python3
"""
_AlCaLumiPixels_

Scenario supporting proton collisions

"""

from Configuration.DataProcessing.Impl.AlCa import AlCa
from Configuration.Eras.Era_Run2_2017_cff import Run2_2017

class AlCaLumiPixels(AlCa):
    def __init__(self):
        AlCa.__init__(self)
        self.eras=Run2_2017
        self.skims=['AlCaPCCZeroBias+AlCaPCCRandom']
    """
    _AlCaLumiPixels_

    Implement configuration building for data processing for proton
    collision data taking

    """


