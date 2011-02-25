#!/usr/bin/env python
"""
_AlCaP0_

Scenario supporting proton collisions

"""

import os
import sys

from Configuration.DataProcessing.Scenario import Scenario
from Configuration.DataProcessing.Utils import stepALCAPRODUCER
import FWCore.ParameterSet.Config as cms
from Configuration.PyReleaseValidation.ConfigBuilder import ConfigBuilder
from Configuration.PyReleaseValidation.ConfigBuilder import Options
from Configuration.PyReleaseValidation.ConfigBuilder import defaultOptions
from Configuration.PyReleaseValidation.ConfigBuilder import installFilteredStream
from Configuration.DataProcessing.RecoTLR import customisePrompt,customiseExpress

class AlCaP0(Scenario):
    """
    _AlCaP0_

    Implement configuration building for data processing for proton
    collision data taking

    """


    def promptReco(self, globalTag, writeTiers = ['ALCARECO'], **args):
        """
        _promptReco_

        Proton collision data taking prompt reco

        """

        skims = ['EcalCalPi0Calib',
                 'EcalCalEtaCalib']
        step = stepALCAPRODUCER(skims)
        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = "pp"
        options.step = step
        options.isMC = False
        options.isData = True
        options.beamspot = None
        options.eventcontent = ','.join(writeTiers)
        options.datatier = ','.join(writeTiers)
        options.magField = 'AutoFromDBCurrent'
        options.conditions = "FrontierConditions_GlobalTag,%s" % globalTag
        options.relval = False
        
        process = cms.Process('RECO')
        cb = ConfigBuilder(options, process = process, with_output = True)

        # Input source
        process.source = cms.Source("PoolSource",
            fileNames = cms.untracked.vstring()
        )
        cb.prepare()

        #add the former top level patches here
        # customisePrompt(process)

        return process


    def expressProcessing(self, globalTag, writeTiers = [], **args):
        """
        _expressProcessing_

        Proton collision data taking express processing

        """

        skims = ['EcalCalPi0Calib',
                 'EcalCalEtaCalib']
        step = stepALCAPRODUCER(skims)
        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = "pp"
        options.step = step
        options.isMC = False
        options.isData = True
        options.beamspot = None
        options.eventcontent = ','.join(writeTiers)
        options.datatier = ','.join(writeTiers)
        options.magField = 'AutoFromDBCurrent'
        options.conditions = "FrontierConditions_GlobalTag,%s" % globalTag
        options.relval = False
        
        process = cms.Process('RECO')
        cb = ConfigBuilder(options, process = process, with_output = True)

        # Input source
        process.source = cms.Source("PoolSource",
            fileNames = cms.untracked.vstring()
        )
        cb.prepare()

        #add the former top level patches here
        customisePrompt(process)
        
        return process


    def alcaSkim(self, skims, **args):
        """
        _alcaSkim_

        AlcaReco processing & skims for proton collisions

        """

        globalTag = None
        if 'globaltag' in args:
            globalTag = args['globaltag']
        
        step = "ALCAOUTPUT:"
        for skim in skims:
          step += (skim+"+")
        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = "pp"
        options.step = step.rstrip('+')
        options.isMC = False
        options.isData = True
        options.beamspot = None
        options.eventcontent = None
        options.relval = None
        if globalTag != None :
            options.conditions = "FrontierConditions_GlobalTag,%s" % globalTag
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
        options.isMC = False
        options.isData = True
        options.beamspot = None
        options.eventcontent = None
        options.name = "EDMtoMEConvert"
        options.conditions = "FrontierConditions_GlobalTag,%s" % globalTag
        options.arguments = ""
        options.evt_type = ""
        options.filein = []
 
        process = cms.Process("HARVESTING")
        process.source = cms.Source("PoolSource")
        configBuilder = ConfigBuilder(options, process = process)
        configBuilder.prepare()

        #
        # customise process for particular job
        #
        process.source.processingMode = cms.untracked.string('RunsAndLumis')
        process.source.fileNames = cms.untracked(cms.vstring())
        process.maxEvents.input = -1
        process.dqmSaver.workflow = datasetName
        process.dqmSaver.saveByLumiSection = 1
        if args.has_key('referenceFile') and args.get('referenceFile', ''):
            process.DQMStore.referenceFileName = \
                                cms.untracked.string(args['referenceFile'])

        return process
