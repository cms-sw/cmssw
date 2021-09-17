#!/usr/bin/env python3
"""
_ppEra_Run2_2017_ppRef_

Scenario supporting proton collisions

"""

import os
import sys

from Configuration.DataProcessing.Reco import Reco
import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run2_2017_ppRef_cff import Run2_2017_ppRef

from Configuration.DataProcessing.Impl.pp import pp

class ppEra_Run2_2017_ppRef(pp):
    def __init__(self):
        pp.__init__(self)
        self.recoSeq=''
        self.cbSc='pp'
        self.addEI=True
        self.eras=Run2_2017_ppRef
        self.promptCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2017_ppRef' ]
        self.expressCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2017_ppRef' ]
        self.visCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2017_ppRef' ]
    """
    _ppEra_Run2_2017_ppRef_

    Implement configuration building for data processing for proton
    collision data taking for Run2

    """
