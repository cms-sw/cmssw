#!/usr/bin/env python
"""
_ppEra_Run2_2017_trackingOnly

Scenario supporting proton collisions and tracking only reconstruction for HP beamspot

"""

import os
import sys

from Configuration.DataProcessing.Reco import Reco
import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run2_2017_cff import Run2_2017

from Configuration.DataProcessing.Impl.pp import pp

class ppEra_Run2_2017_trackingOnly(pp):
    def __init__(self):
        pp.__init__(self)
        self.recoSeq=''
        self.cbSc='pp'
        self.eras=Run2_2017
        self.promptCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2017' ]
        self.expressCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2017' ]
        self.visCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2017' ]
    """
    _ppEra_Run2_2017_trackingOnly

    Implement configuration building for data processing for proton
    collision data taking for Run2, 2017 high performance beamspot

    """
