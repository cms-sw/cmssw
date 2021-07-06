#!/usr/bin/env python3
"""
_trackingOnlyEra_Run2_2018_pp_on_AA

Scenario supporting 2018 heavyIon collisions and tracking only reconstruction for HP beamspot

"""

import os
import sys

from   Configuration.DataProcessing.Impl.trackingOnly import trackingOnly
import FWCore.ParameterSet.Config as cms
from   Configuration.Eras.Era_Run2_2018_pp_on_AA_cff import Run2_2018_pp_on_AA

from   Configuration.DataProcessing.Impl.pp import pp

class trackingOnlyEra_Run2_2018_pp_on_AA(trackingOnly):
    def __init__(self):
        trackingOnly.__init__(self)
        # tracking only RECO is sufficient, to run high performance BS at PCL;
        # some dedicated customization are required, though: customisePostEra_Run2_2018_trackingOnly
        self.isRepacked=True
        self.eras=Run2_2018_pp_on_AA
        #keep post-era parts the same as in the default 2018 era
        self.promptCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2018_2018_pp_on_AA' ]
        self.expressCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2018_pp_on_AA_express_trackingOnly' ]
        self.visCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2018_2018_pp_on_AA' ]

    """
    _trackingOnlyEra_Run2_2018_pp_on_AA

    Implement configuration building for data processing for 2018 heavyIon
    collision data taking for Run2, 2018 high performance beamspot in pp_on_AA data taking

    """
