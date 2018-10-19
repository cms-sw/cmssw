#!/usr/bin/env python
"""
_cosmicsEra_Run2_2018_HI_

Scenario supporting cosmic data taking

"""

import os
import sys

from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
from Configuration.DataProcessing.Impl.cosmics import cosmics

class cosmicsEra_Run2_2018_HI(cosmics):
    def __init__(self):
        cosmics.__init__(self)
        self.eras = Run2_2018
        self.isRepacked = False
        self.promptCustoms = [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2018_cosmics_hybrid' ]
        self.expressCustoms = [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2018_cosmics_hybrid' ]
        self.visCustoms = [ 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2018_cosmics_hybrid' ]
    """
    _cosmicsEra_Run2_2018_HI_

    Implement configuration building for data processing for cosmic
    data taking in Run2, during the 2018 heavy ion data taking period
    (with the strip tracker in hybrid ZS mode)

    """
    def _setRepackedFlag(self,args):
        if not 'repacked' in args:
            args['repacked']= True

    def promptReco(self, globalTag, **args):
        if not 'customs' in args:
            args['customs']= [ ]
        for c in self.promptCustoms:
            args['customs'].append(c)

        if self.isRepacked:
            self._setRepackedFlag(args)

        return super(cosmicsEra_Run2_2018_HI, self).promptReco(globalTag, **args)

    def expressProcessing(self, globalTag, **args):
        if not 'customs' in args:
            args['customs']=[ ]
        for c in self.expressCustoms:
            args['customs'].append(c)

        if self.isRepacked:
            self._setRepackedFlag(args)

        print("Express: ", args)

        return super(cosmicsEra_Run2_2018_HI, self).expressProcessing(globalTag, **args)

    def visualizationProcessing(self, globalTag, **args):
        if not 'customs' in args:
            args['customs']=[ ]
        for c in self.visCustoms:
            args['customs'].append(c)

        if self.isRepacked:
            self._setRepackedFlag(args)

        return super(cosmicsEra_Run2_2018_HI, self).visualizationProcessing(globalTag, **args)
