#!/usr/bin/env python
"""
_HeavyIonsRun2_

Scenario supporting heavy ions collisions

"""

import os
import sys

from Configuration.DataProcessing.Reco import Reco
import FWCore.ParameterSet.Config as cms

class HeavyIonsRun2(Reco):
    def __init__(self):
        self.recoSeq=''
        self.cbSc='HeavyIons'
        self.promptCustoms='Configuration/DataProcessing/RecoTLR.customiseRun2PromptHI'
        self.expressCustoms='Configuration/DataProcessing/RecoTLR.customiseRun2ExpressHI'
        self.visCustoms='Configuration/DataProcessing/RecoTLR.customiseRun2ExpressHI'
    """
    _HeavyIonsRun2_

    Implement configuration building for data processing for Heavy Ions
    collision data taking for Run2

    """

    def _checkMINIAOD(self,**args):
        if 'outputs' in args:
            for a in args['outputs']:
                if a['dataTier'] == 'MINIAOD':
                    raise RuntimeError("MINIAOD is not supported in HeavyIonsRun2")


    def _setRepackedFlag(self,args):
        if not 'repacked' in args:
            args['repacked']= True

    def promptReco(self, globalTag, **args):
        """
        _promptReco_

        Heavy ions collision data taking prompt reco

        """
        self._checkMINIAOD(**args)
        self._setRepackedFlag(args)

        if not 'skims' in args:
            args['skims']=['@allForPrompt']

        customsFunction = self.promptCustoms
        if not 'customs' in args:
            args['customs']=[ customsFunction ]
        else:
            args['customs'].append(customsFunction)

        process = Reco.promptReco(self,globalTag, **args)

        return process


    def expressProcessing(self, globalTag, **args):
        """
        _expressProcessing_

        Heavy ions collision data taking express processing

        """
        self._checkMINIAOD(**args)
        self._setRepackedFlag(args)

        if not 'skims' in args:
            args['skims']=['@allForExpress']

        customsFunction = self.expressCustoms
        if not 'customs' in args:
            args['customs']=[ customsFunction ]
        else:
            args['customs'].append( customsFunction )

        process = Reco.expressProcessing(self,globalTag, **args)
        
        return process

    def visualizationProcessing(self, globalTag, **args):
        """
        _visualizationProcessing_

        Heavy ions collision data taking visualization processing

        """
        self._checkMINIAOD(**args)
        self._setRepackedFlag(args)

        customsFunction = self.visCustoms
        if not 'customs' in args:
            args['customs']=[ customsFunction ]
        else:
            args['customs'].append( customsFunction )

        process = Reco.visualizationProcessing(self,globalTag, **args)
        
        return process

    def alcaHarvesting(self, globalTag, datasetName, **args):
        """
        _alcaHarvesting_

        Heavy ions collisions data taking AlCa Harvesting

        """
        self._checkMINIAOD(**args)

        if not 'skims' in args and not 'alcapromptdataset' in args:
            args['skims']=['BeamSpotByRun',
                           'BeamSpotByLumi',
                           'SiStripQuality']
            
        return Reco.alcaHarvesting(self, globalTag, datasetName, **args)

