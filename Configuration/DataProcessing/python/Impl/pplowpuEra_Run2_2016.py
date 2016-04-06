#!/usr/bin/env python
"""
_pplowpuEra_Run2_2016_

Scenario supporting proton collisions

"""

import os
import sys

from Configuration.DataProcessing.Reco import Reco
import FWCore.ParameterSet.Config as cms
import Configuration.StandardSequences.Eras as eras

from Configuration.DataProcessing.Impl.pplowpu import pplowpu

class pplowpuEra_Run2_2016(pplowpu):
    def __init__(self):
        pplowpu.__init__(self)
        self.recoSeq=''
        self.cbSc='pp'
        self.eras=eras.eras.Run2_2016
        self.promptCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2016' ]
        self.expressCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2016' ]
        self.visCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2016' ]
    """
    _pplowpuEra_Run2_2016_

    Implement configuration building for data processing for proton
    collision data taking for Run2

    """
