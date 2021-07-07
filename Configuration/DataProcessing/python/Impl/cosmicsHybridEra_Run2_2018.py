#!/usr/bin/env python3
"""
_cosmicsHybridEra_Run2_2018_

Scenario supporting cosmic data taking

"""

import os
import sys

from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
from Configuration.DataProcessing.Impl.cosmicsHybrid import cosmicsHybrid

class cosmicsHybridEra_Run2_2018(cosmicsHybrid):
    def __init__(self):
        cosmicsHybrid.__init__(self)
        self.eras = Run2_2018
    """
    _cosmicsHybridEra_Run2_2018_

    Implement configuration building for data processing for cosmic
    data taking in Run2, during the 2018 heavy ion data taking period
    (with the strip tracker in hybrid ZS mode)

    """
