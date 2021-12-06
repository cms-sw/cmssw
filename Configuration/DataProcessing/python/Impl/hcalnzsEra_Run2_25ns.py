#!/usr/bin/env python3
"""
_hcalnzsEra_Run2_25ns_

Scenario supporting proton collisions

"""

import os
import sys

from Configuration.DataProcessing.Impl.hcalnzs import hcalnzs
from Configuration.Eras.Era_Run2_25ns_cff import Run2_25ns

class hcalnzsEra_Run2_25ns(hcalnzs):
    def __init__(self):
        hcalnzs.__init__(self)
        self.recoSeq=':reconstruction_HcalNZS'
        self.cbSc='pp'
        self.addEI=True
        self.eras = Run2_25ns
        self.promptCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_25ns' ]
        self.expressCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_25ns' ]
        self.visCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_25ns' ]

    """
    _hcalnzsEra_Run2_25ns_

    Implement configuration building for data processing for proton
    collision data taking

    """
