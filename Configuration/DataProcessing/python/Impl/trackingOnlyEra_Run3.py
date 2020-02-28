#!/usr/bin/env python
"""
_trackingOnlyEra_Run3

Scenario supporting proton collisions and tracking only reconstruction for HP beamspot

"""

import os
import sys

from   Configuration.DataProcessing.Impl.trackingOnly import trackingOnly
import FWCore.ParameterSet.Config as cms
from   Configuration.Eras.Era_Run3_cff import Run3

from   Configuration.DataProcessing.Impl.pp import pp

class trackingOnlyEra_Run3(trackingOnly):
    def __init__(self):
        trackingOnly.__init__(self)
        # tracking only RECO is sufficient, to run high performance BS at PCL;
        # some dedicated customization are required, though: customisePostEra_Run3_trackingOnly
        self.recoSeq=':reconstruction_trackingOnly'
        self.cbSc='pp'
        self.addEI=False
        self.eras=Run3
        self.promptCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3' ]
        self.expressCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3_express_trackingOnly' ]
        self.visCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3' ]

    """
    _trackingOnlyEra_Run3

    Implement configuration building for data processing for proton
    collision data taking for Run3, 2021 high performance beamspot

    """
