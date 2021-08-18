#!/usr/bin/env python3
"""
_trackingOnlyEra_Run2_2018_highBetaStar

Scenario supporting proton collisions and tracking only reconstruction for HP beamspot

"""

import os
import sys

from   Configuration.DataProcessing.Impl.trackingOnly import trackingOnly
import FWCore.ParameterSet.Config as cms
from   Configuration.Eras.Era_Run2_2018_highBetaStar_cff import Run2_2018_highBetaStar

from   Configuration.DataProcessing.Impl.pp import pp

class trackingOnlyEra_Run2_2018_highBetaStar(trackingOnly):
    def __init__(self):
        trackingOnly.__init__(self)
        # tracking only RECO is sufficient, to run high performance BS at PCL;
        # some dedicated customization are required, though: customisePostEra_Run2_2018_trackingOnly
        self.recoSeq=':reconstruction_trackingOnly'
        self.cbSc='pp'
        self.addEI=False
        self.eras=Run2_2018_highBetaStar
        #keep post-era parts the same as in the default 2018 era
        self.promptCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2018' ]
        self.expressCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2018_express_trackingOnly' ]
        self.visCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2018' ]

    """
    _trackingOnlyEra_Run2_2018_highBetaStar

    Implement configuration building for data processing for proton
    collision data taking for Run2, 2018 high performance beamspot in highBetaStar data taking

    """
