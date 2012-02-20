#!/usr/bin/env python
"""
_cosmics_

Scenario supporting cosmic data taking

"""

import os
import sys

from Configuration.DataProcessing.Reco import Reco
from Configuration.DataProcessing.RecoTLR import customiseCosmicData

class cosmics(Reco):
    """
    _cosmics_

    Implement configuration building for data processing for cosmic
    data taking

    """


    def promptReco(self, globalTag, **args):
        """
        _promptReco_

        Cosmic data taking prompt reco

        """

        skims = ['TkAlBeamHalo',
                 'MuAlBeamHaloOverlaps',
                 'MuAlBeamHalo',
                 'TkAlCosmics0T',
                 'MuAlGlobalCosmics',
                 'MuAlCalIsolatedMu',
                 'HcalCalHOCosmics',
                 'DtCalib',
                 'DtCalibCosmics']
        process = self.promptRecoImp(globalTag, skims, args)

        customiseCosmicData(process)  
        return process


    def expressProcessing(self, globalTag, **args):
        """
        _expressProcessing_

        Cosmic data taking express processing

        """

        skims = ['SiStripCalZeroBias',
                 'MuAlCalIsolatedMu']
        process = self.expressProcessingImp(globalTag, skims, args)

        customiseCosmicData(process)  
        return process
