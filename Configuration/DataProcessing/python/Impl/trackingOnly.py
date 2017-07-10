#!/usr/bin/env python
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
        # some dedicated customization are required, though: customisePostEra_Run2_2017_trackingOnly
        self.recoSeq=':reconstruction_trackingOnly'
        self.cbSc='pp'
        # don't run EI, because only tracking is done
        self.addEI=False
    """
    _trackingOnly_

    Implement configuration building for data processing for proton
    collision data taking

    """

    def expressProcessing(self, globalTag, **args):

        # TkAlMinBias run but hidden to Tier0, in order not to persist it
        if not args.has_key('skims') :
            args['skims']=['TkAlMinBias']
        else :
            if not 'TkAlMinBias' in args['skims'] :
                args['skims'].append('TkAlMinBias')

        # reco sequence is limited to tracking => DQM accordingly
        if not args.has_key('dqmSeq') :
            args['dqmSeq'] = ['DQMOfflineTracking']

        process = pp.expressProcessing(self, globalTag, **args)

        return process

    """
    _ppEra_Run2_2017_trackingOnly

    Implement configuration building for data processing for proton
    collision data taking for Run2, 2017 high performance beamspot

    """
