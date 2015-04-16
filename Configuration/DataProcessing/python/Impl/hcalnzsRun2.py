#!/usr/bin/env python
"""
_hcalnzsRun2_

Scenario supporting proton collisions

"""

import os
import sys

from Configuration.DataProcessing.Impl.pp import pp
from Configuration.DataProcessing.RecoTLR import customisePromptRun2

class hcalnzsRun2(pp):
    def __init__(self):
        self.recoSeq=':reconstruction_HcalNZS'
        self.cbSc='pp'
    """
    _hcalnzsRun2_

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
        
        #add the former top level patches here
        customisePromptRun2(process)
        return process
