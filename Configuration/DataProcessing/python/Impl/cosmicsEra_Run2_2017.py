#!/usr/bin/env python3
"""
_cosmicsEra_Run2_2017_

Scenario supporting cosmic data taking

"""

import os
import sys

from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
from Configuration.DataProcessing.Impl.cosmics import cosmics

class cosmicsEra_Run2_2017(cosmics):
    def __init__(self):
        cosmics.__init__(self)
        self.eras = Run2_2017
    """
    _cosmicsEra_Run2_2017_

    Implement configuration building for data processing for cosmic
    data taking in Run2

    """
