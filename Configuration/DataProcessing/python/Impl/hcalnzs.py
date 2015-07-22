#!/usr/bin/env python
"""
_hcalnzs_

Scenario supporting proton collisions

"""

import os
import sys

from Configuration.DataProcessing.Impl.pp import pp

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
        if not 'skims' in args:
            args['skims']=['HcalCalMinBias']
        process = pp.promptReco(self,globalTag,**args)
        return process
