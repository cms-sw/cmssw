#!/usr/bin/env python3
"""
_ppEra_Run3_2026_
Scenario supporting proton collisions for 2026
"""

import os
import sys

from Configuration.DataProcessing.Reco import Reco
import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_2026_cff import Run3_2026

from Configuration.DataProcessing.Impl.pp import pp

class ppEra_Run3_2026(pp):
    def __init__(self):
        pp.__init__(self)
        self.recoSeq=''
        self.cbSc='pp'
        self.eras=Run3_2026
        self.promptCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3_2026' ]
        self.expressCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3_2026' ]
        self.visCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3_2026' ]
    """
    _ppEra_Run3_2026_
    Implement configuration building for data processing for proton
    collision data taking for Run3_2026
    """
