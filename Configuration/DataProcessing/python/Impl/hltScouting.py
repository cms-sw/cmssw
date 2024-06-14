#!/usr/bin/env python3
"""
_hltScouting_

Scenario supporting proton collisions with input HLT scouting data

"""

from __future__ import print_function

import os
import sys

from Configuration.DataProcessing.Scenario import *
from Configuration.DataProcessing.Utils import stepSKIMPRODUCER, addMonitoring, dictIO, nanoFlavours, gtNameAndConnect
import FWCore.ParameterSet.Config as cms

class hltScouting(Scenario):
    def __init__(self):
        Scenario.__init__(self)
        self.recoSeq = ''
        self.cbSc = 'pp'
        self.isRepacked = False
        self.promptCustoms = ['Configuration/DataProcessing/RecoTLR.customisePrompt']
        self.promptModifiers = cms.ModifierChain()
    """
    _hltScouting_

    Implement configuration building for data processing for proton
    collision data taking with input HLT scouting data
    """

    def promptReco(self, globalTag, **args):
        """
        _promptReco_

        Proton collision data taking prompt reco with input HLT scouting data

        """

        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = self.cbSc
        if 'nThreads' in args:
            options.nThreads = args['nThreads']

        PhysicsSkimStep = ''
        if 'PhysicsSkims' in args:
            PhysicsSkimStep = stepSKIMPRODUCER(args['PhysicsSkims'])

        miniAODStep = ''
        nanoAODStep = ''
        
        if 'outputs' in args:
            print(args['outputs'])
            for a in args['outputs']:
                if a['dataTier'] == 'MINIAOD':
                    miniAODStep = ',PAT'
                if a['dataTier'] in ['NANOAOD', 'NANOEDMAOD']:
                    if "nanoFlavours" in args:
                        nanoAODStep = ',NANO' + nanoFlavours(args['nanoFlavours'])
                    else:
                        nanoAODStep = ',NANO:@PHYS+@L1'        

        if not 'customs' in args:
            args['customs'] = []

        for c in self.promptCustoms:
            args['customs'].append(c)
        options.customisation_file = args['customs']
        
        options.isRepacked = args.get('repacked', self.isRepacked)
        
        options.step = ''
        options.step += self.recoSeq + PhysicsSkimStep
        options.step += miniAODStep + nanoAODStep
        options.step += ',ENDJOB'

        dictIO(options, args)
        options.conditions = gtNameAndConnect(globalTag, args)
        
        process = cms.Process('RECO', cms.ModifierChain(self.eras, self.promptModifiers) )
        cb = ConfigBuilder(options, process = process, with_output = True)

        # Input source
        process.source = cms.Source("PoolSource",
            fileNames = cms.untracked.vstring()
        )

        cb.prepare()

        addMonitoring(process)
        
        return process
