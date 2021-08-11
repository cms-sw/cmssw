#!/usr/bin/env python3
"""
_hcalnzsEra_Run3_

Scenario supporting proton collisions

"""

import os
import sys

from Configuration.DataProcessing.Impl.hcalnzs import hcalnzs
from Configuration.Eras.Era_Run3_cff import Run3

class hcalnzsEra_Run3(hcalnzs):
    def __init__(self):
        hcalnzs.__init__(self)
        self.recoSeq=':reconstruction_HcalNZS'
        self.cbSc='pp'
        self.addEI=True
        self.eras = Run3
        self.promptCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3' ]
        self.expressCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3' ]
        self.visCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3' ]
    """
    _hcalnzsEra_Run3_

    Implement configuration building for data processing for proton
    collision data taking

    """
