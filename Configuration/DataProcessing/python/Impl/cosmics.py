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
        if not 'skims' in args:
            args['skims']= ['@allForPromptCosmics']
        process = Reco.promptReco(self,globalTag, **args)

        customiseCosmicData(process)  
        return process


    def expressProcessing(self, globalTag, **args):
        """
        _expressProcessing_

        Cosmic data taking express processing

        """

        if not 'skims' in args:
            args['skims']= ['@allForExpressCosmics']
        process = Reco.expressProcessing(self,globalTag, **args)

        customiseCosmicData(process)  
        return process


    def alcaHarvesting(self, globalTag, datasetName, **args):
        """
        _alcaHarvesting_

        Proton collisions data taking AlCa Harvesting

        """
        if not 'skims' in args:
            args['skims']=['SiStripQuality']
            
        return Reco.alcaHarvesting(self, globalTag, datasetName, **args)
