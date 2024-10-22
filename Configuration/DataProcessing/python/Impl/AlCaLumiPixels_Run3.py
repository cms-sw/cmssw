#!/usr/bin/env python3
"""
_AlCaLumiPixels_Run3_

Scenario supporting proton collisions

"""

from Configuration.DataProcessing.Impl.AlCa import AlCa
from Configuration.Eras.Era_Run3_cff import Run3

class AlCaLumiPixels_Run3(AlCa):
    def __init__(self):
        AlCa.__init__(self)
        self.eras=Run3
        self.skims=['AlCaPCCZeroBias+RawPCCProducer']
    """
    _AlCaLumiPixels_Run3_

    Implement configuration building for data processing for proton
    collision data taking in Run3

    """


