#!/usr/bin/env python
"""
_trackingnocccEra_Run2_2016_

Scenario supporting proton collisions

"""

import os
import sys

from Configuration.DataProcessing.Impl.trackingnoccc import trackingnoccc
import Configuration.StandardSequences.Eras as eras

class trackingnocccEra_Run2_2016(trackingnoccc):
    def __init__(self):
        trackingnoccc.__init__(self)
        self.recoSeq=':reconstruction_trackingOnly'
        self.cbSc='pp'
        self.eras = eras.eras.Run2_2016

    """
    _trackingnocccEra_Run2_2016_

    Implement configuration building for data processing for proton
    collision data taking

    """
