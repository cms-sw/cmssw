#!/usr/bin/env python3
"""
_pp_

Scenario supporting proton collisions

"""

import os
import sys

from Configuration.DataProcessing.Reco import Reco
import FWCore.ParameterSet.Config as cms
from Configuration.DataProcessing.Modifiers import modifyExpress

class pp(Reco):
    def __init__(self):
        Reco.__init__(self)
        self.recoSeq=''
        self.cbSc='pp'
        self.addEI=True
        self.isRepacked=False
        self.promptCustoms= [ 'Configuration/DataProcessing/RecoTLR.customisePrompt' ]
        self.expressCustoms=[ ]
        self.alcaHarvCustoms=[]
        self.expressModifiers = modifyExpress
        self.visCustoms=[ ]
        self.visModifiers = modifyExpress
    """
    _pp_

    Implement configuration building for data processing for proton
    collision data taking

    """

    def _setRepackedFlag(self,args):
        if not 'repacked' in args:
            args['repacked']= True
            
    def promptReco(self, globalTag, **args):
        """
        _promptReco_

        Proton collision data taking prompt reco

        """
        if not 'skims' in args:
            args['skims']=['@allForPrompt']

        if not 'customs' in args:
            args['customs']= [ ]

        for c in self.promptCustoms:
            args['customs'].append(c)

        if self.isRepacked:
            self._setRepackedFlag(args)

        process = Reco.promptReco(self,globalTag, **args)

        return process

    def expressProcessing(self, globalTag, **args):
        """
        _expressProcessing_

        Proton collision data taking express processing

        """
        if not 'skims' in args:
            args['skims']=['@allForExpress']

        if not 'customs' in args:
            args['customs']=[ ]

        for c in self.expressCustoms:
            args['customs'].append(c)

        if self.isRepacked:
            self._setRepackedFlag(args)
            
        process = Reco.expressProcessing(self,globalTag, **args)
        
        return process

    def visualizationProcessing(self, globalTag, **args):
        """
        _visualizationProcessing_

        Proton collision data taking visualization processing

        """
        if not 'customs' in args:
            args['customs']=[ ]

        for c in self.visCustoms:
            args['customs'].append(c)
            
        if self.isRepacked:
            self._setRepackedFlag(args)

        process = Reco.visualizationProcessing(self,globalTag, **args)
        
        return process

    def alcaHarvesting(self, globalTag, datasetName, **args):
        """
        _alcaHarvesting_

        Proton collisions data taking AlCa Harvesting

        """

        if not 'customs' in args:
            args['customs']=[ ]

        for c in self.alcaHarvCustoms:
            args['customs'].append(c)


        if not 'skims' in args and not 'alcapromptdataset' in args:
            args['skims']=['BeamSpotByRun',
                           'BeamSpotByLumi',
                           'SiStripQuality']
            
        return Reco.alcaHarvesting(self, globalTag, datasetName, **args)
