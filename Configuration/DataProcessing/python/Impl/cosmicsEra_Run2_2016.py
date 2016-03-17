#!/usr/bin/env python
"""
_cosmicsEra_Run2_2016_

Scenario supporting cosmic data taking

"""

import os
import sys

import Configuration.StandardSequences.Eras as eras
from Configuration.DataProcessing.Impl.cosmics import cosmics

class cosmicsEra_Run2_2016(cosmics):
    def __init__(self):
        cosmics.__init__(self)
        self.eras = eras.eras.Run2_2016
    """
    _cosmicsEra_Run2_2016_

    Implement configuration building for data processing for cosmic
    data taking in Run2

    """
