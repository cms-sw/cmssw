#!/usr/bin/env python
"""
_HeavyIonsEra_Run2_2018_

Scenario supporting heavy ions collisions

"""

import os
import sys

from Configuration.DataProcessing.Reco import Reco
import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run2_2018_cff import Run2_2018

from Configuration.DataProcessing.Impl.HeavyIons import HeavyIons

class HeavyIonsEra_Run2_2018(HeavyIons):
    def __init__(self):
        HeavyIons.__init__(self)
        self.eras = Run2_2018
        self.promptCustoms += [ 'Configuration/DataProcessing/RecoTLR.customise_HI_PostEra_Run2_2018' ]
        self.expressCustoms += [ 'Configuration/DataProcessing/RecoTLR.customise_HI_PostEra_Run2_2018' ]
        self.visCustoms += [ 'Configuration/DataProcessing/RecoTLR.customise_HI_PostEra_Run2_2018' ]
    """
    _HeavyIonsEra_Run2_2018_

    Implement configuration building for data processing for Heavy Ions
    collision data taking for Run2 in 2018

    """

