#!/usr/bin/env python3
"""
_hcalnzsEra_Run2_2018_highBetaStar_

Scenario supporting proton collisions

"""

import os
import sys

from Configuration.DataProcessing.Impl.hcalnzs import hcalnzs
from Configuration.Eras.Era_Run2_2018_highBetaStar_cff import Run2_2018_highBetaStar

class hcalnzsEra_Run2_2018_highBetaStar(hcalnzs):
    def __init__(self):
        hcalnzs.__init__(self)
        self.recoSeq=':reconstruction_HcalNZS'
        self.cbSc='pp'
        self.addEI=True
        self.eras = Run2_2018_highBetaStar
        #keep post-era parts the same as in the default 2018 era
        self.promptCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2018' ]
        self.expressCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2018' ]
        self.visCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2018' ]
    """
    _hcalnzsEra_Run2_2018_highBetaStar_

    Implement configuration building for data processing for proton
    collision data taking for Run2, 2018 hcal nzs workflow in highBetaStar data taking

    """
