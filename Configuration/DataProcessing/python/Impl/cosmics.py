#!/usr/bin/env python
"""
_cosmics_

Scenario supporting cosmic data taking

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


class cosmics(Scenario):
    """
    _cosmics_

    Implement configuration building for data processing for cosmic
    data taking

    """


    def promptReco(self, globalTag, writeTiers = ['RECO','ALCARECO']):
        """
        _promptReco_

        Cosmic data taking prompt reco

        """

        skims = ['TkAlBeamHalo',
                 'MuAlBeamHaloOverlaps',
                 'MuAlBeamHalo',
                 'TkAlCosmics0T',
                 'MuAlStandAloneCosmics',
                 'MuAlGlobalCosmics',
                 'MuAlCalIsolatedMu',
                 'HcalCalHOCosmics']
        step = stepALCAPRODUCER(skims)
        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = "cosmics"
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
 
        return process

    def expressProcessing(self, globalTag,  writeTiers = [],
                          datasets = [], alcaDataset = None):
        """
        _expressProcessing_

        Implement Cosmics Express processing

        """

        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = "cosmics"
        options.step = \
          """RAW2DIGI,L1Reco,RECO:reconstructionCosmics,ALCA:MuAlCalIsolatedMu+RpcCalHLT+TkAlCosmicsHLT+TkAlCosmics0T+MuAlStandAloneCosmics+MuAlGlobalCosmics+HcalCalHOCosmics,ENDJOB"""
        options.isMC = False
        options.isData = True
        options.eventcontent = None
        options.relval = None
        options.beamspot = None
        options.conditions = "FrontierConditions_GlobalTag,%s" % globalTag
        
        process = cms.Process('EXPRESS')
        cb = ConfigBuilder(options, process = process)

        process.source = cms.Source(
           "NewEventStreamFileReader",
           fileNames = cms.untracked.vstring()
        )
        
        cb.prepare()

        #  //
        # // Install the OutputModules for everything but ALCA
        #//
        self.addExpressOutputModules(process, writeTiers, datasets)
        
        #  //
        # // TODO: Install Alca output
        #//
        
        return process
    

    def alcaSkim(self, skims):
        """
        _alcaSkim_

        AlcaReco processing & skims for cosmics

        """
        step = "ALCAOUTPUT:"
        for skim in skims:
            step += (skim+"+")
        options = Options()
        options.__dict__.update(defaultOptions.__dict__)
        options.scenario = 'cosmics'        
        options.step = step+'DQM,ENDJOB'
        options.isMC = False
        options.isData = True
        options.beamspot = None
        options.eventcontent = None
        options.relval = None
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
                

        

        


    def dqmHarvesting(self, datasetName, runNumber,  globalTag, **options):
        """
        _dqmHarvesting_

        Cosmic data taking DQM Harvesting

        """
        options = defaultOptions
        options.scenario = "cosmics"
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

        return process
