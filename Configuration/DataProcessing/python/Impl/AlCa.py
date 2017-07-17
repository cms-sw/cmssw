#!/usr/bin/env python
"""
_AlCa_

Scenario supporting proton collisions

"""

import os
import sys

from Configuration.DataProcessing.Scenario import *
from Configuration.DataProcessing.Utils import stepALCAPRODUCER,dqmIOSource,harvestingMode,dictIO,gtNameAndConnect,addMonitoring
import FWCore.ParameterSet.Config as cms

class AlCa(Scenario):
    def __init__(self):
        Scenario.__init__(self)

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

        process = cms.Process('RECO', self.eras)
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
        step = ""
        pclWflws = [x for x in skims if "PromptCalibProd" in x]
        skims = filter(lambda x: x not in pclWflws, skims)

        if len(pclWflws):
            step += 'ALCA:'+('+'.join(pclWflws))

        if len(skims) > 0:
            if step != "":
                step += ","
            step += "ALCAOUTPUT:"+('+'.join(skims))

        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = "pp"
        options.step = step
        options.conditions = args['globaltag'] if 'globaltag' in args else 'None'
        if 'globalTagConnect' in args and args['globalTagConnect'] != '':
            options.conditions += ','+args['globalTagConnect']

        options.triggerResultsProcess = 'RECO'

        process = cms.Process('ALCA', self.eras)
        cb = ConfigBuilder(options, process=process)

        # Input source
        process.source = cms.Source(
           "PoolSource",
           fileNames=cms.untracked.vstring()
        )

        cb.prepare()

        # FIXME: dirty hack..any way around this?
        # Tier0 needs the dataset used for ALCAHARVEST step to be a different data-tier
        for wfl in pclWflws:
            methodToCall = getattr(process, 'ALCARECOStream'+wfl)
            methodToCall.dataset.dataTier = cms.untracked.string('ALCAPROMPT')

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

        process = cms.Process("HARVESTING", self.eras)
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
