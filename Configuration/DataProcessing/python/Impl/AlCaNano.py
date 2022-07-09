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

    def alcaSkim(self, skims, **args):
        """
        _alcaSkim_
        AlcaReco processing & skims for proton collisions
        """
        step = ""
        pclWflws = [x for x in skims if "PromptCalibProd" in x]
        skims = [x for x in skims if x not in pclWflws]

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

        harvestingMode(process,datasetName,args)

        return process

    def alcaHarvesting(self, globalTag, datasetName, **args):
        """
        _alcaHarvesting_
        Proton collisions data taking AlCa Harvesting
        """
        skims = []
        if 'skims' in args:
            skims = args['skims']


        if 'alcapromptdataset' in args:
            skims.append('@'+args['alcapromptdataset'])

        if len(skims) == 0: return None
        options = defaultOptions
        options.scenario = self.cbSc if hasattr(self,'cbSc') else self.__class__.__name__
        options.step = "ALCAHARVEST:"+('+'.join(skims))
        options.name = "ALCAHARVEST"
        options.conditions = gtNameAndConnect(globalTag, args)

        process = cms.Process("ALCAHARVEST", self.eras)
        process.source = cms.Source("PoolSource")

        if 'customs' in args:
            options.customisation_file=args['customs']

        configBuilder = ConfigBuilder(options, process = process)
        configBuilder.prepare()

        process.source.processingMode = cms.untracked.string('RunsAndLumis')
        process.source.fileNames = cms.untracked(cms.vstring())
        process.maxEvents.input = -1
        process.dqmSaver.workflow = datasetName

        return process

    def expressProcessing(self, globalTag, **args):
        """
        _expressProcessing_
        Proton collision data taking express processing
        """
        skims = []
        if 'skims' in args:
            skims = args['skims']
            pclWkflws = [x for x in skims if "PromptCalibProd" in x]
            for wfl in pclWkflws:
                skims.remove(wfl)

        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = "pp"
        options.step = stepALCAPRODUCER(skims)

        if 'outputs' in args:
            # the RAW data-tier needs a special treatment since the event-content as defined in release is not good enough
            outputs_Raw = [x for x in args['outputs'] if x['dataTier'] == 'RAW']
            outputs_noRaw = [x for x in args['outputs'] if x['dataTier'] != 'RAW']
            if len(outputs_Raw) == 1:
                print('RAW data-tier requested')
            options.outputDefinition = outputs_noRaw.__str__()

        options.conditions = gtNameAndConnect(globalTag, args)

        options.filein = 'tobeoverwritten.xyz'
        if 'inputSource' in args:
            options.filetype = args['inputSource']
        process = cms.Process('RECO', self.eras)

        if 'customs' in args:
            options.customisation_file=args['customs']

        cb = ConfigBuilder(options, process = process, with_output = True, with_input = True)

        cb.prepare()

        addMonitoring(process)

        for output in outputs_Raw:
            print(output)
            moduleLabel = output['moduleLabel']
            selectEvents = output.get('selectEvents', None)
            maxSize = output.get('maxSize', None)

            outputModule = cms.OutputModule(
                "PoolOutputModule",
                fileName = cms.untracked.string("%s.root" % moduleLabel)
                )

            outputModule.dataset = cms.untracked.PSet(dataTier = cms.untracked.string("RAW"))

            if maxSize != None:
                outputModule.maxSize = cms.untracked.int32(maxSize)

            if selectEvents != None:
                outputModule.SelectEvents = cms.untracked.PSet(
                    SelectEvents = cms.vstring(selectEvents)
                    )
            outputModule.outputCommands = cms.untracked.vstring('drop *',
                                                                'keep  *_*_*_HLT')

            setattr(process, moduleLabel, outputModule)
	        # outputModule=getattr(self.process,theModuleLabel)
            setattr(process, moduleLabel+'_step', cms.EndPath(outputModule))
            path = getattr(process, moduleLabel+'_step')
            process.schedule.append(path)

        return process

