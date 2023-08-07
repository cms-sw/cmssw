#!/usr/bin/env python3
"""
_ppEra_Run3_pp_on_PbPb_approxSiStripClusters_2023_

Scenario supporting proton collisions

"""

import os
import sys

from Configuration.DataProcessing.Reco import Reco
from Configuration.Eras.Era_Run3_pp_on_PbPb_approxSiStripClusters_2023_cff import Run3_pp_on_PbPb_approxSiStripClusters_2023
import FWCore.ParameterSet.Config as cms

from Configuration.DataProcessing.Impl.pp import pp

class ppEra_Run3_pp_on_PbPb_approxSiStripClusters_2023(pp):
    def __init__(self):
        pp.__init__(self)
        self.recoSeq=''
        self.cbSc='pp'
        self.isRepacked=True
        self.eras=Run3_pp_on_PbPb_approxSiStripClusters_2023
        self.promptCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3_pp_on_PbPb_approxSiStripClusters_2023' ]
        self.expressCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3_pp_on_PbPb_approxSiStripClusters_2023' ]
        self.visCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3_pp_on_PbPb_approxSiStripClusters_2023' ]

    """
    _ppEra_Run3_pp_on_PbPb_approxSiStripClusters_2023_

    Implement configuration building for data processing for pp-like processing of HI
    collision data taking for Run3 with approxSiStripClusters (rawprime format)

    """
