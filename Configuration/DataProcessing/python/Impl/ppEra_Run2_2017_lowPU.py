#!/usr/bin/env python
"""
_ppEra_Run2_2017_lowpU_

Scenario supporting proton collisions

"""

import os
import sys

from Configuration.DataProcessing.Reco import Reco
import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run2_2017_lowPU_cff import Run2_2017_lowPU

from Configuration.DataProcessing.Impl.pp import pp

class ppEra_Run2_2017_lowPU(pp):
    def __init__(self):
        pp.__init__(self)
        self.recoSeq=''
        self.cbSc='pp'
        self.addEI=True
        self.isRepacked=True
        self.eras=Run2_2017_lowPU
        self.promptCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2017_lowPU' ]
        self.expressCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2017_lowPU' ]
        self.visCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2017_lowPU' ]
    """
    _ppEra_Run2_2017_lowPU_

    Implement configuration building for data processing for proton
    collision data taking for Run2

    """
