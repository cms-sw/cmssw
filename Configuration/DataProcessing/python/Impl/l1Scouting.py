#!/usr/bin/env python3
"""
_l1Scouting_

Scenario supporting proton collisions with input L1-Scouting data

"""
from Configuration.DataProcessing.Scenario import *
from Configuration.DataProcessing.Utils import stepSKIMPRODUCER, addMonitoring, dictIO, nanoFlavours, gtNameAndConnect

import FWCore.ParameterSet.Config as cms

class l1Scouting(Scenario):
    def __init__(self):
        Scenario.__init__(self)
        self.recoSeq = ''
        self.cbSc = 'pp'
        self.isRepacked = False
        self.promptCustoms = ['Configuration/DataProcessing/RecoTLR.customisePrompt']
        self.promptModifiers = cms.ModifierChain()
    """
    _l1Scouting_

    Implement configuration building for data processing for proton
    collision data taking with input L1-Scouting data
    """

    def promptReco(self, globalTag, **args):
        """
        _promptReco_

        Proton collision data taking prompt reco with input L1-Scouting data

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
            outputs = []
            for a in args['outputs']:
                if a['dataTier'] in ['NANOAOD', 'NANOEDMAOD']:
                    if 'nanoFlavours' not in args:
                        raise SystemExit(f'l1Scouting: fatal error - requesting {a["dataTier"]} dataTier without specifying a NanoFlavour'
                                          ' ("nanoFlavours" must be either ["@L1Scout"] or ["@L1ScoutSelect"])')
                    args_nanoFlavours = args['nanoFlavours']
                    if args_nanoFlavours not in [['@L1Scout'], ['@L1ScoutSelect']]:
                        raise SystemExit(f'l1Scouting: fatal error - invalid "nanoFlavours": {args_nanoFlavours}'
                                          ' (must be either ["@L1Scout"] or ["@L1ScoutSelect"])')
                    nanoAODStep = ',NANO' + nanoFlavours(args_nanoFlavours)
                    outputs.append(a)
                else:
                    print(f'l1Scouting: warning - dataTier:{a["dataTier"]} is currently not supported and will be removed from outputs')
            if {output['dataTier'] for output in outputs} != {a['dataTier'] for a in args['outputs']}:
                print(f'l1Scouting: warning - the outputs will be changed from {args["outputs"]} to {outputs}')
                args['outputs'] = outputs

        if not 'customs' in args:
            args['customs'] = []

        for c in self.promptCustoms:
            args['customs'].append(c)
        options.customisation_file = args['customs']

        options.isRepacked = args.get('repacked', self.isRepacked)

        options.step = ''
        options.step += self.recoSeq + PhysicsSkimStep
        options.step += miniAODStep + nanoAODStep

        dictIO(options, args)
        options.conditions = gtNameAndConnect(globalTag, args)

        process = cms.Process('L1SCOUT', cms.ModifierChain(self.eras, self.promptModifiers))
        cb = ConfigBuilder(options, process = process, with_output = True)

        # Input source
        process.source = cms.Source("PoolSource",
            fileNames = cms.untracked.vstring()
        )

        cb.prepare()

        addMonitoring(process)

        return process
