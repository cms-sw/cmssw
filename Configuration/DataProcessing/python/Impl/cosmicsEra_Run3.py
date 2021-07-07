#!/usr/bin/env python3
"""
_cosmicsEra_Run3_

Scenario supporting cosmic data taking

"""

import os
import sys

from Configuration.Eras.Era_Run3_cff import Run3
from Configuration.DataProcessing.Impl.cosmics import cosmics

class cosmicsEra_Run3(cosmics):
    def __init__(self):
        cosmics.__init__(self)
        self.eras = Run3
    """
    _cosmicsEra_Run3_

    Implement configuration building for data processing for cosmic
    data taking in Run3

    """
