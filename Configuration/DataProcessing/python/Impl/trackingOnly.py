#!/usr/bin/env python3
"""
_trackingOnly_

Scenario supporting proton collisions and tracking only reconstruction for HP beamspot

"""

import os
import sys

from Configuration.DataProcessing.Impl.pp import pp

class trackingOnly(pp):
    def __init__(self):
        pp.__init__(self)
        # tracking only RECO is sufficient, to run high performance BS at PCL;
        # some dedicated customization are required, though: see specific era implementations
        self.recoSeq=':reconstruction_trackingOnly'
        self.cbSc='pp'
        # don't run EI, because only tracking is done
        self.addEI=False
    """
    _trackingOnly_

    Implement configuration building for data processing for proton
    collision data taking for high performance beamspot

    """

    def expressProcessing(self, globalTag, **args):

        # TkAlMinBias run but hidden to Tier0, in order not to persist it
        if 'skims' not in args :
            args['skims']=['TkAlMinBias']
        else :
            if not 'TkAlMinBias' in args['skims'] :
                args['skims'].append('TkAlMinBias')

        # reco sequence is limited to tracking => DQM accordingly
        if 'dqmSeq' not in args or len(args['dqmSeq'])==0:
            args['dqmSeq'] = ['DQMOfflineTracking']

        process = pp.expressProcessing(self, globalTag, **args)

        return process


