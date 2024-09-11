#!/usr/bin/env python3
"""
_ppEra_Run3_pp_on_PbPb_2024_

Scenario supporting heavy ions collisions

"""

import os
import sys

from Configuration.DataProcessing.Reco import Reco
import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_pp_on_PbPb_2024_cff import Run3_pp_on_PbPb_2024

from Configuration.DataProcessing.Impl.pp import pp

class ppEra_Run3_pp_on_PbPb_2024(pp):
    def __init__(self):
        pp.__init__(self)
        self.recoSeq=''
        self.cbSc='pp'
        self.isRepacked=True
        self.eras=Run3_pp_on_PbPb_2024
        self.promptCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3_pp_on_PbPb_2024' ]
        self.expressCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3_pp_on_PbPb_2024' ]
        self.visCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3_pp_on_PbPb_2024' ]
    """
    _ppEra_Run3_pp_on_PbPb_2024_

    Implement configuration building for data processing for pp-like processing of HI
    collision data taking for Run3

    """
