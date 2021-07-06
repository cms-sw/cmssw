#!/usr/bin/env python3
"""
_hcalnzsEra_Run2_2018_pp_on_AA_

Scenario supporting 2018 heavyIon collisions

"""

import os
import sys

from Configuration.DataProcessing.Impl.hcalnzs import hcalnzs
from Configuration.Eras.Era_Run2_2018_pp_on_AA_cff import Run2_2018_pp_on_AA

class hcalnzsEra_Run2_2018_pp_on_AA(hcalnzs):
    def __init__(self):
        hcalnzs.__init__(self)
        self.isRepacked=True
        self.eras=Run2_2018_pp_on_AA
        #keep post-era parts the same as in the default 2018 era
        self.promptCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2018_pp_on_AA' ]
        self.expressCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2018_pp_on_AA' ]
        self.visCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2018_pp_on_AA' ]
    """
    _hcalnzsEra_Run2_2018_pp_on_AA_

    Implement configuration building for data processing for heavyIon 2018
    collision data taking for Run2, 2018 hcal nzs workflow in pp_on_AA data taking

    """
