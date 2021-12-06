#!/usr/bin/env python3
"""
_ppEra_Run3_

Scenario supporting proton collisions

"""

import os
import sys

from Configuration.DataProcessing.Reco import Reco
import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_cff import Run3

from Configuration.DataProcessing.Impl.pp import pp

class ppEra_Run3(pp):
    def __init__(self):
        pp.__init__(self)
        self.recoSeq=''
        self.cbSc='pp'
        self.addEI=True
        self.eras=Run3
        self.promptCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3' ]
        self.expressCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3' ]
        self.visCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3' ]
    """
    _ppEra_Run3_

    Implement configuration building for data processing for proton
    collision data taking for Run3

    """
