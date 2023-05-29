#!/usr/bin/env python3
"""
_trackingOnlyEra_Run3_pp_on_PbPb

Scenario supporting Run3 heavyIon collisions and tracking only reconstruction for HP beamspot

"""

import os
import sys

from   Configuration.DataProcessing.Impl.trackingOnly import trackingOnly
import FWCore.ParameterSet.Config as cms
from   Configuration.Eras.Era_Run3_pp_on_PbPb_cff import Run3_pp_on_PbPb

from   Configuration.DataProcessing.Impl.pp import pp

class trackingOnlyEra_Run3_pp_on_PbPb(trackingOnly):
    def __init__(self):
        trackingOnly.__init__(self)
        # tracking only RECO is sufficient, to run high performance BS at PCL;
        # some dedicated customization are required, though: customisePostEra_Run2_2018_trackingOnly
        self.isRepacked=True
        self.eras=Run3_pp_on_PbPb
        #keep post-era parts the same as in the default 2018 era
        self.promptCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3_pp_on_PbPb' ]
        self.expressCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3_pp_on_PbPb_express_trackingOnly' ]
        self.visCustoms += [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3_pp_on_PbPb' ]

    """
    _trackingOnlyEra_Run3_pp_on_PbPb

    Implement configuration building for data processing for Run3 heavyIon
    collision data taking for Run3, high performance beamspot in pp_on_PbPb data taking

    """
