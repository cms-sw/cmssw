#!/usr/bin/env python3
"""
_HeavyIons_

Scenario supporting heavy ions collisions

"""

import os
import sys

from Configuration.DataProcessing.Reco import Reco
import FWCore.ParameterSet.Config as cms
from Configuration.DataProcessing.Modifiers import modifyExpressHI

class HeavyIons(Reco):
    def __init__(self):
        Reco.__init__(self)
        self.recoSeq=''
        self.cbSc='HeavyIons'
        self.promptCustoms= [ 'Configuration/DataProcessing/RecoTLR.customisePromptHI' ]
        self.expressCustoms= [ ]
        self.expressModifiers = modifyExpressHI
        self.visCustoms= [ ]
        self.visModifiers = modifyExpressHI
    """
    _HeavyIons_

    Implement configuration building for data processing for Heavy Ions
    collision data taking

    """

    def _checkMINIAOD(self,**args):
        if 'outputs' in args:
            for a in args['outputs']:
                if a['dataTier'] == 'MINIAOD':
                    raise RuntimeError("MINIAOD is not supported in HeavyIons")

                
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

        if not 'customs' in args:
            args['customs']=[ ]

        for c in self.promptCustoms:
            args['customs'].append(c)

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

        if not 'customs' in args:
            args['customs']=[ ]

        for c in self.expressCustoms:
            args['customs'].append(c)

        process = Reco.expressProcessing(self,globalTag, **args)
        
        return process

    def visualizationProcessing(self, globalTag, **args):
        """
        _visualizationProcessing_

        Heavy ions collision data taking visualization processing

        """
        self._checkMINIAOD(**args)
        self._setRepackedFlag(args)

        if not 'customs' in args:
            args['customs']=[ ]

        for c in self.visCustoms:
            args['customs'].append(c)

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

