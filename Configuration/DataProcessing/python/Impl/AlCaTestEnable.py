#!/usr/bin/env python
"""
_AlCaTestEnable_

Scenario supporting proton collisions

"""

from Configuration.DataProcessing.Impl.AlCa import AlCa

class AlCaTestEnable(AlCa):
    def __init__(self):
        self.skims=['TkAlLAS']
    """
    _AlCaTestEnable_

    Implement configuration building for data processing for proton
    collision data taking

    """
