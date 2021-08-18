#!/usr/bin/env python3
"""
_ppEra_Run2_2017_pp_on_XeXe_

Scenario supporting proton collisions

"""

import os
import sys

from Configuration.DataProcessing.Reco import Reco
import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run2_2017_pp_on_XeXe_cff import Run2_2017_pp_on_XeXe

from Configuration.DataProcessing.Impl.pp import pp

class ppEra_Run2_2017_pp_on_XeXe(pp):
    def __init__(self):
        pp.__init__(self)
        self.recoSeq=''
        self.cbSc='pp'
        self.addEI=True
        self.isRepacked=True
        self.eras=Run2_2017_pp_on_XeXe
        self.promptCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2017_pp_on_XeXe' ]
        self.expressCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2017_pp_on_XeXe' ]
        self.visCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2017_pp_on_XeXe' ]
    """
    _ppEra_Run2_2017_pp_on_XeXe_

    Implement configuration building for data processing for proton
    collision data taking for Run2

    """
