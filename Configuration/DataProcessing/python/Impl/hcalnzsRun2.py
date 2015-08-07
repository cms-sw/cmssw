#!/usr/bin/env python
"""
_hcalnzsRun2_

Scenario supporting proton collisions

"""

import os
import sys

from Configuration.DataProcessing.Impl.pp import pp

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

        if not 'customs' in args:
            args['customs']=['Configuration/DataProcessing/RecoTLR.customisePromptRun2']
        else:
            args['customs'].append('Configuration/DataProcessing/RecoTLR.customisePromptRun2')

        process = pp.promptReco(self,globalTag,**args)
        
        return process
