#!/usr/bin/env python3
"""
_ppEra_Run2_50ns_

Scenario supporting proton collisions

"""

import os
import sys

from Configuration.DataProcessing.Reco import Reco
import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run2_50ns_cff import Run2_50ns

from Configuration.DataProcessing.Impl.pp import pp

class ppEra_Run2_50ns(pp):
    def __init__(self):
        pp.__init__(self)
        self.recoSeq=''
        self.cbSc='pp'
        self.addEI=True
        self.eras=Run2_50ns
    """
    _ppEra_Run2_50ns_

    Implement configuration building for data processing for proton
    collision data taking for Run2

    """
