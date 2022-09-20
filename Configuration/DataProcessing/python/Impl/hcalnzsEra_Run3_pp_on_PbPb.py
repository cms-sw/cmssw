#!/usr/bin/env python3
"""
_hcalnzsEra_Run3_pp_on_AA_

Scenario supporting Run3 heavyIon collisions

"""

import os
import sys

from Configuration.DataProcessing.Impl.hcalnzs import hcalnzs
from Configuration.Eras.Era_Run3_pp_on_PbPb_cff import Run3_pp_on_PbPb

class hcalnzsEra_Run3_pp_on_PbPb(hcalnzs):
    def __init__(self):
        hcalnzs.__init__(self)
        self.isRepacked=True
        self.eras=Run3_pp_on_PbPb
        #keep post-era parts the same as in the default Run3 era
        self.promptCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3_pp_on_PbPb' ]
        self.expressCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3_pp_on_PbPb' ]
        self.visCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3_pp_on_PbPb' ]
    """
    _hcalnzsEra_Run3_pp_on_PbPb_

    Implement configuration building for data processing for heavyIon Run3
    collision data taking for Run3 hcal nzs workflow in pp_on_PbPb data taking

    """
