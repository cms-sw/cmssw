#!/usr/bin/env python3
"""
_ppEra_Run3_2025_
Scenario supporting proton collisions for 2025
"""

import os
import sys

from Configuration.DataProcessing.Reco import Reco
import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_2025_cff import Run3_2025

from Configuration.DataProcessing.Impl.pp import pp

class ppEra_Run3_2025(pp):
    def __init__(self):
        pp.__init__(self)
        self.recoSeq=''
        self.cbSc='pp'
        self.eras=Run3_2025
        self.promptCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3_2025' ]
        self.expressCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3_2025' ]
        self.visCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3_2025' ]
    """
    _ppEra_Run3_2025_
    Implement configuration building for data processing for proton
    collision data taking for Run3_2025
    """
