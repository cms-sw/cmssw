#!/usr/bin/env python
"""
_AlCaLumiPixels_

Scenario supporting proton collisions

"""

from Configuration.DataProcessing.Impl.AlCa import AlCa

class AlCaLumiPixels(AlCa):
    def __init__(self):
        self.skims=['LumiPixels']
    """
    _AlCaLumiPixels_

    Implement configuration building for data processing for proton
    collision data taking

    """


