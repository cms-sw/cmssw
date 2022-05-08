#!/usr/bin/env python3
"""
_AlCaNano_

Scenario supporting proton collisions for AlCa needs when ALCANANO is produced

"""
from __future__ import print_function

import os
import sys

from Configuration.DataProcessing.Scenario import *
from Configuration.DataProcessing.Utils import stepALCAPRODUCER,dqmIOSource,harvestingMode,dictIO,gtNameAndConnect,addMonitoring
import FWCore.ParameterSet.Config as cms

class AlCaNano(Scenario):
    def __init__(self):
        Scenario.__init__(self)
        self.recoSeq=''
        self.promptCustoms= [ 'Configuration/DataProcessing/RecoTLR.customisePrompt' ]
        self.promptModifiers = cms.ModifierChain()

    """
    _AlCaNano_

    Implement configuration building for data processing for proton
    collision data taking for AlCa needs when ALCANANO is produced

    """

    def skimsIfNotGiven(self,args,sl):
        if not 'skims' in args:
            args['skims']=sl

    def promptReco(self, globalTag, **args):
        if not 'skims' in args:
            args['skims']=self.skims
        if not 'customs' in args:
            args['customs']= [ ]
        for c in self.promptCustoms:
            args['customs'].append(c)

        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = "pp"
        dictIO(options,args)
        options.conditions = gtNameAndConnect(globalTag, args)

        if 'customs' in args:
            print(args['customs'])
            options.customisation_file=args['customs']

        options.step = 'RECO'
        options.step += self.recoSeq
        options.step += stepALCAPRODUCER(args['skims'])

        process = cms.Process('RECO', cms.ModifierChain(self.eras, self.promptModifiers) )
        cb = ConfigBuilder(options, process = process, with_output = True)

        # Input source
        process.source = cms.Source("PoolSource",
            fileNames = cms.untracked.vstring()
        )
        cb.prepare()

        return process
