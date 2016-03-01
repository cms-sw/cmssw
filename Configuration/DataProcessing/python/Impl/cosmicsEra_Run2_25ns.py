#!/usr/bin/env python
"""
_cosmicsEra_Run2_25ns_

Scenario supporting cosmic data taking

"""

import os
import sys

import Configuration.StandardSequences.Eras as eras
from Configuration.DataProcessing.Impl.cosmics import cosmics

class cosmicsEra_Run2_25ns(cosmics):
    def __init__(self):
        cosmics.__init__(self)
        self.eras = eras.eras.Run2_25ns
    """
    _cosmicsEra_Run2_25ns_

    Implement configuration building for data processing for cosmic
    data taking in Run2

    """
