#!/usr/bin/env python3
"""
_ppEra_Run2_2018_highBetaStar_

Scenario supporting proton collisions

"""

import os
import sys

from Configuration.DataProcessing.Reco import Reco
import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run2_2018_highBetaStar_cff import Run2_2018_highBetaStar

from Configuration.DataProcessing.Impl.pp import pp

class ppEra_Run2_2018_highBetaStar(pp):
    def __init__(self):
        pp.__init__(self)
        self.recoSeq=''
        self.cbSc='pp'
        self.addEI=True
        self.eras=Run2_2018_highBetaStar
        #keep post-era parts the same as in the default 2018 era
        self.promptCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2018' ]
        self.expressCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2018' ]
        self.visCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2018' ]
    """
    _ppEra_Run2_2018_highBetaStar_

    Implement configuration building for data processing for proton
    collision data taking for Run2 2018 highBetaStar data taking

    """
