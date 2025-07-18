#!/usr/bin/env python3
"""
_ppEra_Run3_2024_ppRef_
Scenario supporting ppRef collisions for 2024
"""

import os
import sys

from Configuration.DataProcessing.Reco import Reco
import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_2024_ppRef_cff import Run3_2024_ppRef

from Configuration.DataProcessing.Impl.pp import pp

class ppEra_Run3_2024_ppRef(pp):
    def __init__(self):
        pp.__init__(self)
        self.recoSeq=''
        self.cbSc='pp'
        self.eras=Run3_2024_ppRef
        self.promptCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3_2024_ppRef' ]
        self.expressCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3_2024_ppRef' ]
        self.visCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3_2024_ppRef' ]
    """
    _ppEra_Run3_2024_ppRef_
    Implement configuration building for data processing for proton
    collision data taking for Run3_2024
    """
