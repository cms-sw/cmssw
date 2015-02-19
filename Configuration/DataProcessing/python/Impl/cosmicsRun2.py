#!/usr/bin/env python
"""
_cosmicsRun2_

Scenario supporting cosmic data taking

"""

import os
import sys

from Configuration.DataProcessing.Reco import Reco
from Configuration.DataProcessing.RecoTLR import customiseCosmicDataRun2

class cosmicsRun2(Reco):
    def __init__(self):
        self.recoSeq=''
        self.cbSc='cosmics'
    """
    _cosmicsRun2_

    Implement configuration building for data processing for cosmic
    data taking in Run2

    """


    def promptReco(self, globalTag, **args):
        """
        _promptReco_

        Cosmic data taking prompt reco

        """
        if not 'skims' in args:
            args['skims']= ['@allForPromptCosmics']
        process = Reco.promptReco(self,globalTag, **args)

        customiseCosmicDataRun2(process)  
        return process


    def expressProcessing(self, globalTag, **args):
        """
        _expressProcessing_

        Cosmic data taking express processing

        """

        if not 'skims' in args:
            args['skims']= ['@allForExpressCosmics']
        process = Reco.expressProcessing(self,globalTag, **args)

        customiseCosmicDataRun2(process)  
        return process

    def visualizationProcessing(self, globalTag, **args):
        """
        _visualizationProcessing_

        Cosmic data taking visualization processing

        """

        process = Reco.visualizationProcessing(self,globalTag, **args)

        customiseCosmicDataRun2(process)  
        process.reconstructionCosmics.remove(process.lumiProducer)

        return process

    def alcaHarvesting(self, globalTag, datasetName, **args):
        """
        _alcaHarvesting_

        Proton collisions data taking AlCa Harvesting

        """

        if not 'skims' in args and not 'alcapromptdataset' in args:
            args['skims']=['SiStripQuality']
            
        return Reco.alcaHarvesting(self, globalTag, datasetName, **args)
