#!/usr/bin/env python
"""
_AlCa_

Scenario supporting proton collisions

"""

import os
import sys

from Configuration.DataProcessing.Scenario import *
from Configuration.DataProcessing.Utils import stepALCAPRODUCER,dqmIOSource,harvestingMode,dictIO,gtNameAndConnect
import FWCore.ParameterSet.Config as cms

class AlCa(Scenario):
    """
    _AlCa_

    Implement configuration building for data processing for proton
    collision data taking

    """

    def skimsIfNotGiven(self,args,sl):
        if not 'skims' in args:
            args['skims']=sl

    def promptReco(self, globalTag, **args):
        if not 'skims' in args:
            args['skims']=self.skims
        step = stepALCAPRODUCER(args['skims'])
        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = "pp"
        options.step = step
        dictIO(options,args)
        options.conditions = gtNameAndConnect(globalTag, args)
        
        process = cms.Process('RECO')
        cb = ConfigBuilder(options, process = process, with_output = True)

        # Input source
        process.source = cms.Source("PoolSource",
            fileNames = cms.untracked.vstring()
        )
        cb.prepare()

        return process

    def alcaSkim(self, skims, **args):
        """
        _alcaSkim_

        AlcaReco processing & skims for proton collisions

        """
        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = "pp"
        options.step = "ALCAOUTPUT:"+('+'.join(skims))
        options.conditions = args['globaltag'] if 'globaltag' in args else 'None'
        if args.has_key('globalTagConnect') and args['globalTagConnect'] != '':
            options.conditions += ','+args['globalTagConnect']

        options.triggerResultsProcess = 'RECO'
        
        process = cms.Process('ALCA')
        cb = ConfigBuilder(options, process = process)

        # Input source
        process.source = cms.Source(
           "PoolSource",
           fileNames = cms.untracked.vstring()
        )

        cb.prepare() 

        return process


    def dqmHarvesting(self, datasetName, runNumber, globalTag, **args):
        """
        _dqmHarvesting_

        Proton collisions data taking DQM Harvesting

        """
        options = defaultOptions
        options.scenario = "pp"
        options.step = "HARVESTING:alcaHarvesting"
        options.name = "EDMtoMEConvert"
        options.conditions = gtNameAndConnect(globalTag, args)
 
        process = cms.Process("HARVESTING")
        process.source = dqmIOSource(args)
        configBuilder = ConfigBuilder(options, process = process)
        configBuilder.prepare()

        #
        # customise process for particular job
        #
        #process.source.processingMode = cms.untracked.string('RunsAndLumis')
        #process.source.fileNames = cms.untracked(cms.vstring())
        #process.maxEvents.input = -1
        #process.dqmSaver.workflow = datasetName
        #process.dqmSaver.saveByLumiSection = 1
        #if args.has_key('referenceFile') and args.get('referenceFile', ''):
        #    process.DQMStore.referenceFileName = \
        #                        cms.untracked.string(args['referenceFile'])
        harvestingMode(process,datasetName,args)
        
        return process
