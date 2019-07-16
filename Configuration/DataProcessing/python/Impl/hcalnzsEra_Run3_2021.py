#!/usr/bin/env python
"""
_hcalnzsEra_Run3_2021_

Scenario supporting proton collisions

"""

import os
import sys

from Configuration.DataProcessing.Impl.hcalnzs import hcalnzs
from Configuration.Eras.Era_Run3_cff import Run3

class hcalnzsEra_Run3_2021(hcalnzs):
    def __init__(self):
        hcalnzs.__init__(self)
        self.recoSeq=':reconstruction_HcalNZS'
        self.cbSc='pp'
        self.addEI=True
        self.eras = Run3
        self.promptCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3_2021' ]
        self.expressCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3_2021' ]
        self.visCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3_2021' ]
    """
    _hcalnzsEra_Run3_2021_

    Implement configuration building for data processing for proton
    collision data taking

    """
