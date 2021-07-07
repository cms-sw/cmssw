#!/usr/bin/env python3
"""
_hcalnzsEra_Run2_2018_

Scenario supporting proton collisions

"""

import os
import sys

from Configuration.DataProcessing.Impl.hcalnzs import hcalnzs
from Configuration.Eras.Era_Run2_2018_cff import Run2_2018

class hcalnzsEra_Run2_2018(hcalnzs):
    def __init__(self):
        hcalnzs.__init__(self)
        self.recoSeq=':reconstruction_HcalNZS'
        self.cbSc='pp'
        self.addEI=True
        self.eras = Run2_2018
        self.promptCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2018' ]
        self.expressCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2018' ]
        self.visCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2018' ]
    """
    _hcalnzsEra_Run2_2018_

    Implement configuration building for data processing for proton
    collision data taking

    """
