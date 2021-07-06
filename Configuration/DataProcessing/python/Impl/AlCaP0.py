#!/usr/bin/env python3
"""
_AlCaP0_

Scenario supporting proton collisions

"""

from Configuration.DataProcessing.Impl.AlCa import AlCa

class AlCaP0(AlCa):
    def __init__(self):
        AlCa.__init__(self)
        self.skims=['@AlCaP0']
    """
    _AlCaP0_

    Implement configuration building for data processing for proton
    collision data taking

    """


