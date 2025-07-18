#!/usr/bin/env python3
"""
_ppEra_Run3_2025_OXY_
Scenario supporting OXY collisions for 2025
"""

import os
import sys

from Configuration.DataProcessing.Reco import Reco
import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_2025_OXY_cff import Run3_2025_OXY

from Configuration.DataProcessing.Impl.pp import pp

class ppEra_Run3_2025_OXY(pp):
    def __init__(self):
        pp.__init__(self)
        self.recoSeq=''
        self.cbSc='pp'
        self.isRepacked=True
        self.eras=Run3_2025_OXY
        self.promptCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3_2025_OXY' ]
        self.expressCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3_2025_OXY' ]
        self.visCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3_2025_OXY' ]
    """
    _ppEra_Run3_2025_OXY_
    Implement configuration building for data processing for proton
    collision data taking for Run3_2025
    """
