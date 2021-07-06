#!/usr/bin/env python3
"""
_ppEra_Run2_2018_pp_on_AA_

Scenario supporting proton collisions

"""

import os
import sys

from Configuration.DataProcessing.Reco import Reco
import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run2_2018_pp_on_AA_cff import Run2_2018_pp_on_AA

from Configuration.DataProcessing.Impl.pp import pp

class ppEra_Run2_2018_pp_on_AA(pp):
    def __init__(self):
        pp.__init__(self)
        self.recoSeq=''
        self.cbSc='pp'
        self.addEI=True
        self.isRepacked=True
        self.eras=Run2_2018_pp_on_AA
        self.promptCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2018_pp_on_AA' ]
        self.expressCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2018_pp_on_AA' ]
        self.visCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2018_pp_on_AA' ]
    """
    _ppEra_Run2_2018_pp_on_AA_

    Implement configuration building for data processing for pp-like processing of HI
    collision data taking for Run2 in 2018

    """
