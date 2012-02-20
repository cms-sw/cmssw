#!/usr/bin/env python
"""
_hcalnzs_

Scenario supporting proton collisions

"""

import os
import sys

from Configuration.DataProcessing.Impl.pp import pp
from Configuration.DataProcessing.RecoTLR import customisePrompt

class hcalnzs(pp):
    def __init__(self):
        self.recoSeq=':reconstruction_HcalNZS'
        self.cbSc='pp'
    """
    _hcalnzs_

    Implement configuration building for data processing for proton
    collision data taking

    """
    def promptReco(self, globalTag, **args):
        """
        _promptReco_

        Proton collision data taking prompt reco

        """
        skims=['HcalCalMinBias']
        process = self.promptRecoImp(self,globalTag, skims, args)
        customisePrompt(process)
        return process
