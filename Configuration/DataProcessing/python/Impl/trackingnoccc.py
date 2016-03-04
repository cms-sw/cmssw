#!/usr/bin/env python
"""
_trackingnoccc_

Scenario supporting proton collisions

"""

import os
import sys

from Configuration.DataProcessing.Impl.pp import pp

class trackingnoccc(pp):
    def __init__(self):
        pp.__init__(self)
        self.recoSeq=':reconstruction_trackingOnly'
        self.cbSc='pp'
        self.promptCustoms  += [ 'RecoTracker/Configuration/customiseForRunIInoCCC.customiseForRunIInoCCC' ]
        self.expressCustoms += [ 'RecoTracker/Configuration/customiseForRunIInoCCC.customiseForRunIInoCCC' ]

    """
    _trackingnoccc_

    Implement configuration building for data processing for proton
    collision data taking without CCC in the tracking sequence

    """
    def promptReco(self, globalTag, **args):
        """
        _promptReco_

        Proton collision data taking prompt reco

        """

#        if not 'skims' in args:
#            args['skims']=['SiStripAlCaMinBias']

        process = pp.promptReco(self,globalTag,**args)

        return process
