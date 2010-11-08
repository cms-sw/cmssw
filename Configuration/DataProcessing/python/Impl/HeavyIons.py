#!/usr/bin/env python
"""
_HeavyIons_

Scenario supporting heavy-ion collisions

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
from Configuration.PyReleaseValidation.ConfigBuilder import addOutputModule
from Configuration.DataProcessing.RecoTLR import customisePromptHI,customiseExpressHI

class HeavyIons(Scenario):
    """
    _HeavyIons_

    Implement configuration building for data processing for 
    heavy-ion collision data taking

    """


    def promptReco(self, globalTag, writeTiers = ['RECO'], **args):
        """
        _promptReco_

        Heavy-ion collision data taking prompt reco

        """

        skims = ['SiStripCalZeroBias']
        step = stepALCAPRODUCER(skims)
        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = "HeavyIons"
        options.step = 'RAW2DIGI,L1Reco,RECO'+step+',L1HwVal,DQM,ENDJOB'
        options.isMC = False
        options.isData = True
        options.beamspot = None
        options.eventcontent = None
        options.magField = 'AutoFromDBCurrent'
        options.conditions = "FrontierConditions_GlobalTag,%s" % globalTag
        options.relval = False
        
        process = cms.Process('RECO')
        cb = ConfigBuilder(options, process = process)

        # Input source
        process.source = cms.Source("PoolSource",
            fileNames = cms.untracked.vstring()
        )
        cb.prepare()

        for tier in writeTiers: 
          addOutputModule(process, tier, tier)
          
        #add the former top level patches here
        customisePromptHI(process)
        
        return process


    def expressProcessing(self, globalTag, writeTiers = [], **args):
        """
        _expressProcessing_

        Heavy-ion collision data taking express processing

        """

        skims = ['SiStripCalZeroBias']
        step = stepALCAPRODUCER(skims)
        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = "HeavyIons"
        options.step = 'RAW2DIGI,L1Reco,RECO'+step+',L1HwVal,DQM,ENDJOB'
        options.isMC = False
        options.isData = True
        options.beamspot = None
        options.eventcontent = None
        options.magField = 'AutoFromDBCurrent'
        options.conditions = "FrontierConditions_GlobalTag,%s" % globalTag
        options.relval = False
        
        process = cms.Process('RECO')
        cb = ConfigBuilder(options, process = process)

        # Input source
        process.source = cms.Source("NewEventStreamFileReader",
            fileNames = cms.untracked.vstring()
        )
        cb.prepare() 

        for tier in writeTiers: 
          addOutputModule(process, tier, tier)
          
        #add the former top level patches here
        customiseExpressHI(process)
        
        return process


    def alcaSkim(self, skims, **args):
        """
        _alcaSkim_

        AlcaReco processing & skims for heavy-ion collisions

        """

        globalTag = None
        if 'globaltag' in args:
            globalTag = args['globaltag']

        step = ""
        if 'PromptCalibProd' in skims:
            step = "ALCA:PromptCalibProd" 
            skims.remove('PromptCalibProd')
        
        if len( skims ) > 0:
            if step != "":
                step += ","
            step += "ALCAOUTPUT:"
                
        for skim in skims:
          step += (skim+"+")
        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = "HeavyIons"
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

        # FIXME: dirty hack..any way around this?
        # Tier0 needs the dataset used for ALCAHARVEST step to be a different data-tier
        if 'PromptCalibProd' in step:
            process.ALCARECOStreamPromptCalibProd.dataset.dataTier = cms.untracked.string('ALCAPROMPT')

        return process


    def dqmHarvesting(self, datasetName, runNumber, globalTag, **args):
        """
        _dqmHarvesting_

        Heavy-ion collisions data taking DQM Harvesting

        """
        options = defaultOptions
        options.scenario = "HeavyIons"
        options.step = "HARVESTING:dqmHarvesting"
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


    def alcaHarvesting(self, globalTag, **args):
        """
        _alcaHarvesting_

        Heavy-ion collisions data taking AlCa Harvesting

        """
        options = defaultOptions
        options.scenario = "HeavyIons"
        options.step = "ALCAHARVEST:BeamSpotByRun+BeamSpotByLumi"
        options.isMC = False
        options.isData = True
        options.beamspot = None
        options.eventcontent = None
        options.name = "ALCAHARVEST"
        options.conditions = globalTag
        options.arguments = ""
        options.evt_type = ""
        options.filein = []
 
        process = cms.Process("ALCAHARVEST")
        process.source = cms.Source("PoolSource")
        configBuilder = ConfigBuilder(options, process = process)
        configBuilder.prepare()

        #
        # customise process for particular job
        #
        process.source.processingMode = cms.untracked.string('RunsAndLumis')
        process.source.fileNames = cms.untracked(cms.vstring())
        process.maxEvents.input = -1

        return process

