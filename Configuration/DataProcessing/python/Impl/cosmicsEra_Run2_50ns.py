#!/usr/bin/env python3
"""
_cosmicsEra_Run2_50ns_

Scenario supporting cosmic data taking

"""

import os
import sys

from Configuration.Eras.Era_Run2_50ns_cff import Run2_50ns
from Configuration.DataProcessing.Impl.cosmics import cosmics

class cosmicsEra_Run2_50ns(cosmics):
    def __init__(self):
        cosmics.__init__(self)
        self.eras = Run2_50ns
    """
    _cosmicsEra_Run2_50ns_

    Implement configuration building for data processing for cosmic
    data taking in Run2

    """
