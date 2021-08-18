#!/usr/bin/env python3
"""
_HeavyIonsEra_Run2_HI_

Scenario supporting heavy ions collisions

"""

import os
import sys

from Configuration.DataProcessing.Reco import Reco
import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run2_HI_cff import Run2_HI

from Configuration.DataProcessing.Impl.HeavyIons import HeavyIons

class HeavyIonsEra_Run2_HI(HeavyIons):
    def __init__(self):
        HeavyIons.__init__(self)
        self.eras = Run2_HI
    """
    _HeavyIonsEra_Run2_HI_

    Implement configuration building for data processing for Heavy Ions
    collision data taking for Run2

    """

