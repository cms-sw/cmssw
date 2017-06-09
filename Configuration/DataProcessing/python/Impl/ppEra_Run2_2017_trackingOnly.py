#!/usr/bin/env python
"""
_ppEra_Run2_2017_trackingOnly

Scenario supporting proton collisions and tracking only reconstruction for HP beamspot

"""

import os
import sys

from Configuration.DataProcessing.Reco import Reco
import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run2_2017_cff import Run2_2017

from Configuration.DataProcessing.Impl.pp import pp

class ppEra_Run2_2017_trackingOnly(pp):
    def __init__(self):
        pp.__init__(self)
        self.recoSeq=':reconstruction_trackingOnly'
        self.cbSc='pp'
        self.eras=Run2_2017
        self.promptCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2017' ]
        self.expressCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2017' ]
        self.visCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2017' ]

    def expressProcessing(self, globalTag, **args):

        # TkAlMinBias run but hidden to Tier0, in order not to persist it
        if not args.has_key('skims') :
            args['skims'] = 'TkAlMinBias'
        else :
            if not 'TkAlMinBias' in args['skims'] :
                args['skims'].append('TkAlMinBias')

        # reco sequence is limited to tracking => DQM accordingly
        if not args.has_key('dqmSeq') :
            args['dqmSeq'] = 'DQMOfflineTracking'


        pp.expressProcessing(self, globalTag, **args)

    """
    _ppEra_Run2_2017_trackingOnly

    Implement configuration building for data processing for proton
    collision data taking for Run2, 2017 high performance beamspot

    """
