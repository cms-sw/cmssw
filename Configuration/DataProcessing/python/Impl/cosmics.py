#!/usr/bin/env python
"""
_cosmics_

Scenario supporting cosmic data taking

"""

import os
import sys

from Configuration.DataProcessing.Reco import Reco

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

        if not 'customs' in args:
            args['customs']=['Configuration/DataProcessing/RecoTLR.customiseCosmicData']
        else:
            args['customs'].append('Configuration/DataProcessing/RecoTLR.customiseCosmicData')

        process = Reco.promptReco(self,globalTag, **args)

        return process


    def expressProcessing(self, globalTag, **args):
        """
        _expressProcessing_

        Cosmic data taking express processing

        """

        if not 'skims' in args:
            args['skims']= ['@allForExpressCosmics']

        if not 'customs' in args:
            args['customs']=['Configuration/DataProcessing/RecoTLR.customiseCosmicData']
        else:
            args['customs'].append('Configuration/DataProcessing/RecoTLR.customiseCosmicData')
        process = Reco.expressProcessing(self,globalTag, **args)

        return process

    def visualizationProcessing(self, globalTag, **args):
        """
        _visualizationProcessing_

        Cosmic data taking visualization processing

        """

        if not 'customs' in args:
            args['customs']=['Configuration/DataProcessing/RecoTLR.customiseCosmicData']
        else:
            args['customs'].append('Configuration/DataProcessing/RecoTLR.customiseCosmicData')
        process = Reco.visualizationProcessing(self,globalTag, **args)

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
